//===-- ELFObjectFileAnalysisPass.hpp -------------------------------------===//
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
/// This file describes <tt>ELFObjectFileAnalysisPass</tt>,
/// which provides a reference to the ELF object file being lifted to other
/// lifting passes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_ELF_OBJECT_FILE_ANALYSIS_PASS_HPP
#define LUTHIER_TOOLING_COMMON_ELF_OBJECT_FILE_ANALYSIS_PASS_HPP
#include <llvm/IR/PassManager.h>
#include <llvm/Object/ELFObjectFile.h>

namespace luthier {

/// \brief Analysis pass used to provide access to the ELF object file being
/// lifted
class ELFObjectFileAnalysisPass
    : public llvm::AnalysisInfoMixin<ELFObjectFileAnalysisPass> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

  const llvm::object::ELFObjectFileBase &ObjFile;

public:
  class Result {
    friend class ELFObjectFileAnalysisPass;

    const llvm::object::ELFObjectFileBase &ObjFile;

    // NOLINTBEGIN(google-explicit-constructor)
    /*implicit*/ Result(const llvm::object::ELFObjectFileBase &ObjFile)
        : ObjFile(ObjFile) {}
    // NOLINTEND(google-explicit-constructor)

  public:
    [[nodiscard]] const llvm::object::ELFObjectFileBase &getObject() const {
      return ObjFile;
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// constructor
  explicit ELFObjectFileAnalysisPass(llvm::object::ELFObjectFileBase &ObjFile)
      : ObjFile(ObjFile) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) { return ObjFile; }
};

} // namespace luthier

#endif