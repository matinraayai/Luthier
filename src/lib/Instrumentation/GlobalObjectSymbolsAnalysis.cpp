//===-- GlobalObjectSymbolsAnalysis.cpp ---------------------------*- C++ -*-===//
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
/// Implements the \c GlobalValueSymbolsAnalysis class.
//===----------------------------------------------------------------------===//
#include <luthier/Instrumentation/GlobalObjectSymbolsAnalysis.h>
#include <luthier/Instrumentation/ObjectFileAnalysisPass.h>

namespace luthier {

llvm::AnalysisKey GlobalObjectSymbolsAnalysis::Key;

std::optional<llvm::object::SymbolRef>
GlobalObjectSymbolsAnalysis::Result::getSymbolRef(
    const llvm::GlobalObject &GO) const {
  auto It = GlobalObjectSymbols.find(GO);
  return It != GlobalObjectSymbols.end() ? It->second : std::nullopt;
}
} // namespace luthier