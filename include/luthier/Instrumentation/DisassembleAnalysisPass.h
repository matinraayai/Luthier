//===-- DisassembleAnalysisPass.h -------------------------------*- C++ -*-===//
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
///
/// \file
/// Describes the <tt>DisassembleAnalysisPass</tt>, used to disassemble the
/// contents of a function symbol in an object file.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_DISASSEMBLE_ANALYSIS_PASS_H
#define LUTHIER_INSTRUMENTATION_DISASSEMBLE_ANALYSIS_PASS_H
#include <llvm/IR/PassManager.h>
#include <luthier/Instrumentation/Instr.h>

namespace luthier {

/// \brief Function analysis used to disassemble the contents of the function's
/// symbol
class DisassemblerAnalysisPass
    : public llvm::AnalysisInfoMixin<DisassemblerAnalysisPass> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class DisassemblerAnalysisPass;

    const std::vector<Instr> Instructions;

    // NOLINTBEGIN(google-explicit-constructor)
    /*implicit*/ Result(std::vector<Instr> Instructions)
        : Instructions(std::move(Instructions)) {}
    // NOLINTEND(google-explicit-constructor)

  public:
    [[nodiscard]] llvm::ArrayRef<Instr> getInstructions() const {
      return Instructions;
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
               llvm::FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit DisassemblerAnalysisPass() = default;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace luthier

#endif