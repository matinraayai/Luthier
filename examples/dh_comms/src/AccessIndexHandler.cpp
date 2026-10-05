//===-- AccessIndexHandler.cpp - CPU side: aggregate memory indices -------===//
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
#include "dh_comms/AccessIndexHandler.h"

#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::dh_comms {

void AccessIndexHandler::handle(const WaveHeader &Header,
                                llvm::ArrayRef<AccessRecord> Records) {
  Stats &S = PerInstr[Header.InstrId];
  ++S.Waves;
  S.LanesPerWave.add(uint32_t(Records.size()));
  for (const AccessRecord &R : Records) {
    ++S.Lanes;
    if (R.BufferId == UnresolvedBuffer) {
      ++S.Unresolved;
      continue;
    }
    if (R.BufferId < 32)
      S.BufferMask |= 1u << R.BufferId;
    S.Index.add(R.ElementIndex);
    S.Row.add(R.ElementIndex / RowLength);
    S.Col.add(R.ElementIndex % RowLength);
  }
}

static std::string describeAccess(uint32_t Info) {
  const char *Kind[] = {"?", "read", "write", "rmw"};
  return llvm::formatv("{0} {1}B", Kind[accessRwKind(Info)],
                       accessWidthBytes(Info));
}

static std::string describeBuffers(uint32_t Mask) {
  if (Mask == 0)
    return "-";
  std::string Out;
  for (uint32_t I = 0; I < 32; ++I)
    if (Mask & (1u << I))
      Out += (Out.empty() ? "arg" : ",arg") + std::to_string(I);
  return Out;
}

static std::string describeRange(uint32_t Min, uint32_t Max) {
  return llvm::formatv("[{0}, {1}]", Min, Max);
}

bool AccessIndexHandler::report(
    llvm::raw_ostream &OS, llvm::ArrayRef<InstrumentedAccess> Accesses) const {
  OS << llvm::formatv("{0,3}  {1,-22} {2,-9} {3,-6} {4,6} {5,9} {6,10} "
                      "{7,20} {8,13} {9,13}  {10}\n",
                      "id", "instruction", "access", "buffer", "waves",
                      "lanes/wv", "lanes", "linear index", "row", "col",
                      "in range");
  bool AllOk = true;
  for (const auto &[Id, S] : PerInstr) {
    const InstrumentedAccess *A = Id < Accesses.size() ? &Accesses[Id] : nullptr;
    const bool Ok = S.Unresolved == 0 && !S.Index.empty() &&
                    S.Row.Max < NumRows && S.Col.Max < RowLength;
    AllOk &= Ok;
    OS << llvm::formatv(
        "{0,3}  {1,-22} {2,-9} {3,-6} {4,6} {5,9} {6,10} {7,20} {8,13} {9,13}"
        "  {10}\n",
        Id, A ? A->Opcode : "?", A ? describeAccess(A->AccessInfo) : "?",
        describeBuffers(S.BufferMask), S.Waves,
        S.LanesPerWave.Min == S.LanesPerWave.Max
            ? std::to_string(S.LanesPerWave.Min)
            : llvm::formatv("{0}-{1}", S.LanesPerWave.Min, S.LanesPerWave.Max)
                  .str(),
        S.Lanes,
        S.Index.empty() ? "-" : describeRange(S.Index.Min, S.Index.Max),
        S.Row.empty() ? "-" : describeRange(S.Row.Min, S.Row.Max),
        S.Col.empty() ? "-" : describeRange(S.Col.Min, S.Col.Max),
        Ok ? "yes"
           : (S.Unresolved
                  ? llvm::formatv("NO ({0} unresolved)", S.Unresolved).str()
                  : std::string("NO")));
  }
  if (PerInstr.empty())
    OS << "  (no messages received)\n";
  return AllOk && !PerInstr.empty();
}

} // namespace luthier::dh_comms
