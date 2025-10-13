//===-- InstructionTracesAnalysis.h -------------------------------*-
// C++-*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// \file
/// Describes the \c InstructionTracesAnalysis class which provides access to
/// the list of instruction traces discovered during the lifting process.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_DISCOVERED_TRACES_ANALYSIS_H
#define LUTHIER_TOOLING_DISCOVERED_TRACES_ANALYSIS_H
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCInst.h>
#include <map>

namespace luthier {

class TraceInstr {
private:
  /// The MC representation of the instruction
  const llvm::MCInst Inst;
  /// The address on the GPU Agent this instruction is loaded at
  const uint64_t LoadedDeviceAddress;
  /// Size of the instruction
  const size_t Size;

public:
  /// Constructor
  /// \param Inst \c MCInst of the instruction
  /// \param DeviceAddr the device address this instruction is loaded on
  /// \param Size size of the instruction in bytes
  TraceInstr(llvm::MCInst Inst, uint64_t DeviceAddr, size_t Size)
      : Inst(Inst), LoadedDeviceAddress(DeviceAddr), Size(Size) {};

  /// \return the MC representation of the instruction
  [[nodiscard]] llvm::MCInst getMCInst() const { return Inst; }

  /// \return the loaded address of this instruction on the device
  [[nodiscard]] uint64_t getLoadedDeviceAddress() const { LoadedDeviceAddress; }

  /// \return the size of the instruction in bytes
  [[nodiscard]] size_t getSize() const { return Size; }
};

class InstructionTracesAnalysis
    : public llvm::AnalysisInfoMixin<InstructionTracesAnalysis> {
  friend llvm::AnalysisInfoMixin<InstructionTracesAnalysis>;

  static llvm::AnalysisKey Key;

  using Trace = std::map<uint64_t, TraceInstr>;

  using TraceMap = llvm::DenseMap<uint64_t, Trace>;

  /// Mapping between the loaded address of a trace, and the trace;
  /// The trace itself is an ordered map between instruction address and the
  /// trace instructions present in the trace
  TraceMap Traces;

public:
  class Result {
    friend InstructionTracesAnalysis;

    TraceMap &Traces;

    explicit Result(TraceMap &Traces) : Traces(Traces) {};

  public:
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    void insert(uint64_t TraceAddress, Trace &&Trace) {
      (void)Traces.insert({TraceAddress, std::move(Trace)});
    }

    TraceMap::const_iterator begin() const { return Traces.begin(); }

    TraceMap::iterator begin() { return Traces.begin(); }

    TraceMap::const_iterator end() const { return Traces.end(); }

    TraceMap::iterator end() { return Traces.end(); }

    unsigned size() const { return Traces.size(); }

    bool empty() const { return Traces.empty(); }

    bool contains(uint64_t TraceAddr) const {
      return Traces.contains(TraceAddr);
    }

    TraceMap::iterator find(uint64_t TraceAddr) const {
      return Traces.find(TraceAddr);
    }

    TraceMap::iterator find(uint64_t TraceAddr) {
      return Traces.find(TraceAddr);
    }

    Trace operator[](uint64_t TraceAddr) { return Traces[TraceAddr]; }
  };

  InstructionTracesAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{Traces};
  };
};

} // namespace luthier

#endif