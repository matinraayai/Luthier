//===-- CoalescingHandler.cpp - CPU side: count uncoalesced accesses ------===//
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
#include "dh_comms/CoalescingHandler.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>

namespace luthier::dh_comms {

void CoalescingHandler::beginDispatch(const DeviceDescriptor &D) {
  ElementSizeLog2 = D.ElementSizeLog2;
  BaseBySlot.fill(0);
  for (uint32_t I = 0; I < D.NumTrackedBuffers; ++I)
    if (D.BufferArgSlot[I] < MaxTrackedBuffers)
      BaseBySlot[D.BufferArgSlot[I]] = D.BufferBase[I];
}

void CoalescingHandler::handle(const WaveHeader &Header,
                               llvm::ArrayRef<AccessRecord> Records) {
  Stats &S = PerInstr[Header.InstrId];
  ++S.Waves;
  // Bytes each lane moves; an unknown width counts as one dword.
  uint64_t Width = accessWidthBytes(Header.UserData);
  if (Width == 0)
    Width = 4;

  // Lines touched and bytes moved by each group of lanes issued together.
  constexpr unsigned MaxGroups = 64;
  llvm::SmallVector<uint64_t, 16> GroupLines[MaxGroups];
  uint64_t GroupBytes[MaxGroups] = {};

  // The i-th record belongs to the i-th set bit of Exec.
  uint64_t Exec = Header.Exec;
  for (const AccessRecord &R : Records) {
    const unsigned Lane = unsigned(__builtin_ctzll(Exec));
    Exec &= Exec - 1;
    if (R.BufferId == UnresolvedBuffer || R.BufferId >= MaxTrackedBuffers) {
      ++S.Unresolved;
      continue;
    }
    const uint64_t Addr =
        BaseBySlot[R.BufferId] + (uint64_t(R.ElementIndex) << ElementSizeLog2);
    const unsigned Group = Lane / LanesPerCycle;
    GroupBytes[Group] += Width;
    for (uint64_t Line = Addr / CacheLineBytes;
         Line <= (Addr + Width - 1) / CacheLineBytes; ++Line)
      GroupLines[Group].push_back(Line);
  }

  bool Wasted = false;
  for (unsigned G = 0; G < MaxGroups; ++G) {
    if (GroupBytes[G] == 0)
      continue;
    auto &L = GroupLines[G];
    std::sort(L.begin(), L.end());
    const uint64_t Lines = std::unique(L.begin(), L.end()) - L.begin();
    const uint64_t Ideal = (GroupBytes[G] + CacheLineBytes - 1) / CacheLineBytes;
    S.Lines += Lines;
    S.IdealLines += Ideal;
    Wasted |= Lines > Ideal;
  }
  if (Wasted)
    ++S.Uncoalesced;
}

void CoalescingHandler::report(
    llvm::raw_ostream &OS, llvm::ArrayRef<InstrumentedAccess> Accesses) const {
  OS << llvm::formatv("[dh_comms] ===== coalescing ({0}-lane groups, {1}-byte "
                      "lines) =====\n",
                      LanesPerCycle, CacheLineBytes);
  OS << llvm::formatv("{0,3}  {1,-22} {2,10} {3,12} {4,10} {5,12}\n", "id",
                      "instruction", "waves", "uncoalesced", "lines/wv",
                      "ideal/wv");
  uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, Ideal = 0;
  for (const auto &[Id, S] : PerInstr) {
    const InstrumentedAccess *A = Id < Accesses.size() ? &Accesses[Id] : nullptr;
    OS << llvm::formatv("{0,3}  {1,-22} {2,10} {3,12} {4,10:F2} {5,12:F2}\n",
                        Id, A ? A->Opcode : "?", S.Waves, S.Uncoalesced,
                        double(S.Lines) / double(S.Waves),
                        double(S.IdealLines) / double(S.Waves));
    Waves += S.Waves;
    Uncoalesced += S.Uncoalesced;
    Lines += S.Lines;
    Ideal += S.IdealLines;
  }
  OS << llvm::formatv(
      "[dh_comms] VMEM wave-instructions: {0}  coalesced: {1}  uncoalesced: "
      "{2}\n",
      Waves, Waves - Uncoalesced, Uncoalesced);
  OS << llvm::formatv("[dh_comms] cache lines: {0} touched, {1} needed "
                      "({2} wasted)\n",
                      Lines, Ideal, Lines - Ideal);
  OS << "[dh_comms] compare: SQ_INSTS_VMEM = wave-instructions, "
        "SQ_INSTS_VMEM - TD_COALESCABLE_WAVEFRONT_sum = uncoalesced\n";
}

} // namespace luthier::dh_comms
