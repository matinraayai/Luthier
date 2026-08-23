//===-- InitialEntryPointAnalysis.h -------------------------------*-C++-*-===//
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
/// \file InitialEntryPointAnalysis.h
/// Describes the \c InitialEntryPointAnalysis class which provides access to
/// the initial entrypoint of the lifting process, plus the target-module
/// metadata the entry point is recorded in.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INITIAL_ENTRY_POINT_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_INITIAL_ENTRY_POINT_ANALYSIS_H
#include "luthier/ToolCodeGen/EntryPoint.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>

namespace luthier {

/// \brief Name of the target module's named metadata recording the initial
/// entry point of the lifting process.
///
/// \details The node holds a single operand of the form
/// \code !{i64 <raw address>, i1 <is kernel descriptor>} \endcode
/// The address is a loaded (host-visible) address; whoever loads the code
/// object decides how a user-facing entry-point spec maps onto one. Recording
/// it on the module rather than resolving it on demand is what keeps
/// \c InitialEntryPointAnalysis independent of any particular loader, and lets
/// the entry point survive a \c .luthier round-trip.
inline constexpr llvm::StringRef InitialEntryPointMDName =
    "luthier.initial_entry_point";

/// Records \p EP as the initial entry point of \p M, replacing any previously
/// recorded entry point.
void setInitialEntryPoint(llvm::Module &M, const EntryPoint &EP);

/// \return the initial entry point recorded on \p M, or an error if \p M
/// carries no \c luthier.initial_entry_point metadata or the node is malformed
llvm::Expected<EntryPoint> getInitialEntryPoint(const llvm::Module &M);

/// \brief Module analysis exposing the initial entry point recorded on the
/// target module by \c setInitialEntryPoint.
class InitialEntryPointAnalysis
    : public llvm::AnalysisInfoMixin<InitialEntryPointAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend InitialEntryPointAnalysis;

    EntryPoint InitialEP;

    explicit Result(EntryPoint EP) : InitialEP(EP) {};

  public:
    EntryPoint getInitialEntryPoint() const { return InitialEP; }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  InitialEntryPointAnalysis() = default;

  /// Parses \c luthier.initial_entry_point off \p M. A missing or malformed
  /// node is reported on \p M 's context and yields a default \c EntryPoint.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif