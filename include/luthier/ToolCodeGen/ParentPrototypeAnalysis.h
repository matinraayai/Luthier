//===-- ParentPrototypeAnalysis.h ---------------------*- C++ -*-===//
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
/// \file
/// Describes \c ParentPrototypeAnalysis, a module analysis that
/// resolves an \c llvm::Module — either the target module or the
/// instrumentation module — back to the \c Prototype that owns it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PARENT_PROTOTYPE_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_PARENT_PROTOTYPE_ANALYSIS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief External registry that maps every \c llvm::Module owned by a live
/// \c Prototype (both its target module and its instrumentation
/// module) to that prototype.
///
/// The registry is owned by whoever manages the pipeline (e.g. the
/// \c InstrumentationPMDriver) and can hold many prototypes at once. Callers
/// must register a prototype's modules before any pipeline queries the parent
/// via \c ParentPrototypeAnalysis, and unregister them before the
/// prototype (and hence its modules) are destroyed.
class ModuleToPrototypeMap {
  llvm::DenseMap<const llvm::Module *, Prototype *> ModuleToIP;

public:
  ModuleToPrototypeMap() = default;

  ModuleToPrototypeMap(const ModuleToPrototypeMap &) =
      delete;
  ModuleToPrototypeMap &
  operator=(const ModuleToPrototypeMap &) = delete;

  /// Register both modules of \p IP as owned by \p IP.
  void registerPrototype(Prototype &IP);

  /// Remove both modules of \p IP from the registry. Safe to call on a
  /// prototype that was never registered.
  void unregisterPrototype(Prototype &IP);

  /// \return the \c Prototype that owns \p M, or \c nullptr if
  /// \p M has not been registered with this map.
  [[nodiscard]] Prototype *lookup(const llvm::Module &M) const;
};

/// \brief Module-level new-PM analysis that resolves an \c llvm::Module to its
/// parent \c Prototype through a shared
/// \c ModuleToPrototypeMap.
class ParentPrototypeAnalysis
    : public llvm::AnalysisInfoMixin<ParentPrototypeAnalysis> {
  friend llvm::AnalysisInfoMixin<ParentPrototypeAnalysis>;

  static llvm::AnalysisKey Key;

  const ModuleToPrototypeMap &Map;

public:
  class Result {
    friend ParentPrototypeAnalysis;

    Prototype *IP;

    explicit Result(Prototype *IP) : IP(IP) {}

  public:
    /// The parent relationship of a module is fixed for the module's
    /// lifetime, so this analysis result never invalidates regardless of the
    /// \c PreservedAnalyses set reported by other passes.
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    /// \return the parent \c Prototype of the queried module, or
    /// \c nullptr if the module was not registered with the underlying map.
    [[nodiscard]] Prototype *getPrototype() const {
      return IP;
    }
  };

  explicit ParentPrototypeAnalysis(
      const ModuleToPrototypeMap &Map)
      : Map(Map) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    return Result{Map.lookup(M)};
  }
};

} // namespace luthier

#endif
