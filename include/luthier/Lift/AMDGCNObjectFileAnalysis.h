//===-- AMDGCNObjectFileAnalysis.h ------------------------------*- C++ -*-===//
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
/// Describes the \c AMDGCNObjectFileAnalysis class, which provides access to
/// the AMDGCN object file of the lifted code, if available.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LIFT_AMDGCN_OBJECT_FILE_ANALYSIS_H
#define LUTHIER_LIFT_AMDGCN_OBJECT_FILE_ANALYSIS_H
#include "luthier/Object/AMDGCNObjectFile.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Analysis pass used to provide access to the target AMDGCN object
/// file being lifted
class AMDGCNObjectFileAnalysis
    : public llvm::AnalysisInfoMixin<AMDGCNObjectFileAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

  const object::AMDGCNObjectFile &ObjFile;

public:
  class Result {
    friend class AMDGCNObjectFileAnalysis;

    const object::AMDGCNObjectFile &ObjFile;

    explicit Result(const object::AMDGCNObjectFile &ObjFile)
        : ObjFile(ObjFile) {}

  public:
    [[nodiscard]] const object::AMDGCNObjectFile &getObjectFile() const {
      return ObjFile;
    }

    /// Prevents invalidation of the analysis result as the object file doesn't
    /// change throughout the process
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// constructor
  explicit AMDGCNObjectFileAnalysis(const object::AMDGCNObjectFile &ObjFile)
      : ObjFile(ObjFile) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{ObjFile};
  }
};

} // namespace luthier

#endif