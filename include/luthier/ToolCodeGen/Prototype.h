//===-- Prototype.h -----------------------------------*- C++ -*-===//
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
/// Defines \c Prototype, an IR unit representing the state of an
/// ongoing instrumentation task, together with the pass/analysis-manager
/// machinery needed to run passes over them under LLVM's new Pass Manager.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PROTOTYPE_H
#define LUTHIER_TOOL_CODE_GEN_PROTOTYPE_H
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassManagerInternal.h>

namespace llvm {
class PassInstrumentationCallbacks;
} // namespace llvm

namespace luthier {

class Prototype;

using PrototypeAnalysisManager = llvm::AnalysisManager<Prototype>;

using PrototypePassManager = llvm::PassManager<Prototype>;

class Prototype {
  /// Contains the code for the application being instrumented
  std::unique_ptr<llvm::Module> TargetModule{};

  /// Contains the instrumentation logic
  std::unique_ptr<llvm::Module> IModule{};

public:
  explicit Prototype(llvm::StringRef TargetModuleID, llvm::LLVMContext &C)
      : TargetModule(std::make_unique<llvm::Module>(TargetModuleID, C)),
        IModule(std::make_unique<llvm::Module>(
            (llvm::Twine(TargetModuleID) + ".instrumentation_module").str(),
            C)) {};

  Prototype(std::unique_ptr<llvm::Module> Target,
            std::unique_ptr<llvm::Module> IModule);

  Prototype(const Prototype &) = delete;
  Prototype &operator=(const Prototype &) = delete;

  [[nodiscard]] const llvm::Module &getInstrumentationModule() const {
    return *IModule;
  }

  llvm::Module &getInstrumentationModule() { return *IModule; }

  [[nodiscard]] const llvm::Module &getTargetModule() const {
    return *TargetModule;
  }

  llvm::Module &getTargetModule() { return *TargetModule; }

  [[nodiscard]] llvm::StringRef getName() const {
    return TargetModule->getName();
  }

  /// \brief Invokes \p Fn on every <tt>llvm::MachineFunction</tt> of the target
  /// module.
  ///
  /// \details A prototype does not own the target MIR directly: \c
  /// CodeDiscoveryPass lifts it into \c llvm::MachineFunctionAnalysis results
  /// cached on the \c llvm::FunctionAnalysisManager shared by both of the
  /// prototype's modules, keyed by the target module's
  /// <tt>llvm::Function</tt>s. Target functions without a cached result have no
  /// MIR lifted for them and are skipped, so calling this before code discovery
  /// has run is a no-op rather than an error.
  void forEachTargetMF(PrototypeAnalysisManager &PAM,
                       llvm::function_ref<void(llvm::MachineFunction &)> Fn);
};

//===----------------------------------------------------------------------===//
// Cross-level analysis-manager proxies
//===----------------------------------------------------------------------===//

using ModuleAnalysisManagerPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::ModuleAnalysisManager,
                                    Prototype>;

using PrototypeAnalysisManagerModuleProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::Module>;

using FunctionAnalysisManagerPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager,
                                    Prototype>;

using MachineFunctionAnalysisManagerPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager,
                                    Prototype>;

using PrototypeAnalysisManagerFunctionProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::Function>;

using PrototypeAnalysisManagerMachineFunctionProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::MachineFunction>;

/// \brief Marks the three inner analysis-manager proxies as preserved on \p PA.
///
/// \details Every Prototype-level pass that reports anything short of
/// \c PreservedAnalyses::all() must call this unless it genuinely wants the
/// inner managers emptied. The proxies' invalidation hooks call
/// \c InnerAM->clear() — for the whole manager, not just the pass's own module
/// (see the specializations in Prototype.cpp) — which throws away every cached
/// \c MachineFunctionAnalysis result. Those results own the
/// <tt>MachineFunction</tt>s, so dropping them silently replaces the lifted
/// target MIR (and any MIR parsed out of a \c .luthier file) with freshly
/// created, empty <tt>MachineFunction</tt>s for the next pass that asks.
///
/// Per-module and per-function invalidation still happens: a pass that mutates
/// one module reconciles that module's manager directly (as the adaptors in
/// this file do), and analyses at the Prototype level are unaffected by this
/// call.
void preserveInnerAnalysisManagerProxies(llvm::PreservedAnalyses &PA);

/// \brief Adaptor that runs a single \c llvm::Module pass over the target
/// module of an \c Prototype.
///
/// Modeled on LLVM's \c ModuleToFunctionPassAdaptor: instead of owning a whole
/// pass manager, it type-erases and stores one module pass (a
/// \c llvm::ModulePassManager is itself a valid module pass, so an entire
/// pipeline can still be wrapped). \c run runs the pass instrumentation
/// callbacks around the pass and reconciles the inner \c ModuleAnalysisManager
/// before returning.
class RunOnTargetModuleAdaptor
    : public llvm::PassInfoMixin<RunOnTargetModuleAdaptor> {
public:
  using PassConceptT =
      llvm::detail::PassConcept<llvm::Module, llvm::ModuleAnalysisManager>;

  explicit RunOnTargetModuleAdaptor(std::unique_ptr<PassConceptT> Pass)
      : Pass(std::move(Pass)) {}

  llvm::PreservedAnalyses run(Prototype &IP,
                              PrototypeAnalysisManager &IPAM);

  void printPipeline(
      llvm::raw_ostream &OS,
      llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<PassConceptT> Pass;
};

/// \brief Adaptor that runs a single \c llvm::Module pass over the
/// instrumentation module of an \c Prototype.
///
/// See \c RunOnTargetModuleAdaptor for the design rationale.
class RunOnInstrumentationModuleAdaptor
    : public llvm::PassInfoMixin<RunOnInstrumentationModuleAdaptor> {
public:
  using PassConceptT =
      llvm::detail::PassConcept<llvm::Module, llvm::ModuleAnalysisManager>;

  explicit RunOnInstrumentationModuleAdaptor(std::unique_ptr<PassConceptT> Pass)
      : Pass(std::move(Pass)) {}

  llvm::PreservedAnalyses run(Prototype &IP,
                              PrototypeAnalysisManager &IPAM);

  void printPipeline(
      llvm::raw_ostream &OS,
      llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<PassConceptT> Pass;
};

/// Deduce a module pass type and wrap it in a \c RunOnTargetModuleAdaptor.
template <typename ModulePassT>
RunOnTargetModuleAdaptor createRunOnTargetModuleAdaptor(ModulePassT &&Pass) {
  using PassModelT = llvm::detail::PassModel<llvm::Module, ModulePassT,
                                             llvm::ModuleAnalysisManager>;
  // Do not use make_unique, it causes too many template instantiations,
  // causing terrible compile times.
  return RunOnTargetModuleAdaptor(
      std::unique_ptr<RunOnTargetModuleAdaptor::PassConceptT>(
          new PassModelT(std::forward<ModulePassT>(Pass))));
}

/// Deduce a module pass type and wrap it in a
/// \c RunOnInstrumentationModuleAdaptor.
template <typename ModulePassT>
RunOnInstrumentationModuleAdaptor
createRunOnInstrumentationModuleAdaptor(ModulePassT &&Pass) {
  using PassModelT = llvm::detail::PassModel<llvm::Module, ModulePassT,
                                             llvm::ModuleAnalysisManager>;
  // Do not use make_unique, it causes too many template instantiations,
  // causing terrible compile times.
  return RunOnInstrumentationModuleAdaptor(
      std::unique_ptr<RunOnInstrumentationModuleAdaptor::PassConceptT>(
          new PassModelT(std::forward<ModulePassT>(Pass))));
}

} // namespace luthier

//===----------------------------------------------------------------------===//
// Explicit specializations of the inner proxies' invalidation hooks.
//===----------------------------------------------------------------------===//
//
// These MUST live in \c namespace llvm: an explicit specialization of a member
// has to be declared in a namespace enclosing the specialized template.
namespace llvm {

template <>
bool luthier::ModuleAnalysisManagerPrototypeProxy::Result::invalidate(
    luthier::Prototype &IP, const PreservedAnalyses &PA,
    luthier::PrototypeAnalysisManager::Invalidator &Inv);

template <>
bool luthier::FunctionAnalysisManagerPrototypeProxy::Result::
    invalidate(luthier::Prototype &IP, const PreservedAnalyses &PA,
               luthier::PrototypeAnalysisManager::Invalidator &Inv);

template <>
bool luthier::MachineFunctionAnalysisManagerPrototypeProxy::Result::
    invalidate(luthier::Prototype &IP, const PreservedAnalyses &PA,
               luthier::PrototypeAnalysisManager::Invalidator &Inv);

extern template class PassManager<luthier::Prototype>;
extern template class AnalysisManager<luthier::Prototype>;

} // namespace llvm

#endif
