//===-- ELFRelocationResolverPass.hpp -------------------------------------===//
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
/// Describes the \c ELFRelocationResolverAnalysis class, which
/// resolves queries regarding relocation information present inside the
/// ELF object file being lifted.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_ELF_RELOCATION_RESOLVER_ANALYSIS_PASS_H
#define LUTHIER_INSTRUMENTATION_ELF_RELOCATION_RESOLVER_ANALYSIS_PASS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Object/ELFObjectFile.h>

namespace luthier {

/// \brief Analysis pass used to parse relocation information from the
/// ELF object file being lifted and provide that information to other lifting
/// passes
class ELFRelocationResolverAnalysis
    : public llvm::AnalysisInfoMixin<ELFRelocationResolverAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class ELFRelocationResolverAnalysis;

    using RelocationMap =
        llvm::DenseMap<uint64_t, std::pair<const llvm::GlobalObject &,
                                           llvm::object::ELFRelocationRef>>;

    /// Mapping between [object symbol, offset from start of the symbol] ->
    /// relocation section ref of the offset
    RelocationMap Relocations{};

    Result() = default;

  public:
    using const_iterator = RelocationMap::const_iterator;

    const_iterator begin() const { return Relocations.begin(); }

    const_iterator end() const { return Relocations.end(); }

    [[nodiscard]] bool empty() const { return Relocations.empty(); }

    [[nodiscard]] unsigned size() const { return Relocations.size(); }

    const_iterator find(uint64_t Offset) const {
      return Relocations.find(Offset);
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// Default constructor
  ELFRelocationResolverAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif