//===-- InitialExecutionPointAnalysis.h ---------------------------*-C++-*-===//
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
/// \file InitialExecutionPointAnalysis.h
/// Describes the \c InitialExecutionPointAnalysis class which provides access
/// to the initial kernel where the initial entry point was launched.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INITIAL_EXECUTION_POINT_H
#define LUTHIER_TOOL_CODE_GEN_INITIAL_EXECUTION_POINT_H
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>

namespace luthier {

/// \brief Name of the target module's named metadata recording the initial
/// execution point — the kernel the initial entry point was launched from.
///
/// \details The node holds a single operand of the form
/// \code !{i64 <kernel descriptor address>} \endcode
/// See \c InitialEntryPointMDName for why this lives on the module rather than
/// being resolved on demand.
inline constexpr llvm::StringRef InitialExecutionPointMDName =
    "luthier.initial_execution_point";

/// Records \p KD as the initial execution point of \p M, replacing any
/// previously recorded execution point.
void setInitialExecutionPoint(llvm::Module &M,
                              const llvm::amdhsa::kernel_descriptor_t &KD);

/// \return the initial execution point recorded on \p M, or an error if \p M
/// carries no \c luthier.initial_execution_point metadata or it is malformed
llvm::Expected<const llvm::amdhsa::kernel_descriptor_t *>
getInitialExecutionPoint(const llvm::Module &M);

class InitialExecutionPointAnalysis
    : public llvm::AnalysisInfoMixin<InitialExecutionPointAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend InitialExecutionPointAnalysis;

    const llvm::amdhsa::kernel_descriptor_t *InitialExecutionPoint;

    explicit Result(const llvm::amdhsa::kernel_descriptor_t *EP)
        : InitialExecutionPoint(EP) {};

  public:
    /// \return the recorded execution point. Only valid when the module carried
    /// well-formed metadata; a parse failure is reported on the module's context
    /// by \c run and leaves this null.
    [[nodiscard]] const llvm::amdhsa::kernel_descriptor_t &
    getInitialExecutionPoint() const {
      assert(InitialExecutionPoint &&
             "queried the initial execution point of a module that does not "
             "record one");
      return *InitialExecutionPoint;
    }

    [[nodiscard]] bool hasInitialExecutionPoint() const {
      return InitialExecutionPoint != nullptr;
    }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  InitialExecutionPointAnalysis() = default;

  /// Parses \c luthier.initial_execution_point off \p M. A missing or malformed
  /// node is reported on \p M 's context and yields an empty result.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif