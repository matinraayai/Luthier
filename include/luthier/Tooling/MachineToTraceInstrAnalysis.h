//===-- MachineToTraceInstrAnalysis.h -------------------------*- C++-*-===//
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
/// Describes the \c MetadataParserAnalysis class which provides access to a
/// metadata parser to other instrumentation passes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MACHINE_TO_TRACE_INSTR_ANALYSIS_H
#define LUTHIER_TOOLING_MACHINE_TO_TRACE_INSTR_ANALYSIS_H
#include "luthier/Tooling/InstructionTracesAnalysis.h"
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class MachineToTraceInstrAnalysis
    : public llvm::AnalysisInfoMixin<MachineToTraceInstrAnalysis> {
  friend llvm::AnalysisInfoMixin<MachineToTraceInstrAnalysis>;

  static llvm::AnalysisKey Key;

  llvm::DenseMap<const llvm::MachineInstr *, const TraceInstr *>
      MIToTraceInstrMap;

public:
  class Result {
    friend MachineToTraceInstrAnalysis;

    llvm::DenseMap<const llvm::MachineInstr *, const TraceInstr *>
        &MIToTraceInstrMap;

    explicit Result(llvm::DenseMap<const llvm::MachineInstr *,
                                   const TraceInstr *> &MIToTraceInstrMap)
        : MIToTraceInstrMap(MIToTraceInstrMap) {};

  public:
    bool invalidate(llvm::MachineFunction &, const llvm::PreservedAnalyses &,
                    llvm::MachineFunctionAnalysisManager::Invalidator &) {
      return false;
    }

    llvm::DenseMap<const llvm::MachineInstr *, const TraceInstr *> &getMap() {
      return MIToTraceInstrMap;
    }

    const llvm::DenseMap<const llvm::MachineInstr *, const TraceInstr *> &
    getMap() const {
      return MIToTraceInstrMap;
    }
  };

  MachineToTraceInstrAnalysis() = default;

  Result run(llvm::MachineFunction &, llvm::ModuleAnalysisManager &) {
    return Result{MIToTraceInstrMap};
  };
};

} // namespace luthier

#endif