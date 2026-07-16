//===-- InstrumentPrototype.h -----------------------------------*- C++ -*-===//
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
/// Defines \c InstrumentPrototype, an IR unit representing the state of an
/// ongoing instrumentation task, together with the pass/analysis-manager
/// machinery needed to run passes over them under LLVM's new Pass Manager.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INSTRUMENT_PROTOTYPE_H
#define LUTHIER_TOOL_CODE_GEN_INSTRUMENT_PROTOTYPE_H
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

class InstrumentPrototype {
  /// Contains the code for the application being instrumented
  std::unique_ptr<llvm::Module> TargetModule{};

  /// Contains the instrumentation logic
  std::unique_ptr<llvm::Module> IModule{};

public:
  explicit InstrumentPrototype(llvm::StringRef TargetModuleID,
                               llvm::LLVMContext &C)
      : TargetModule(std::make_unique<llvm::Module>(TargetModuleID, C)),
        IModule(std::make_unique<llvm::Module>("instrumentation_module", C)) {};

  InstrumentPrototype(const InstrumentPrototype &) = delete;
  InstrumentPrototype &operator=(const InstrumentPrototype &) = delete;

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
};

using InstrumentPrototypeAnalysisManager =
    llvm::AnalysisManager<InstrumentPrototype>;

using InstrumentPrototypePassManager = llvm::PassManager<InstrumentPrototype>;

//===----------------------------------------------------------------------===//
// Cross-level analysis-manager proxies
//===----------------------------------------------------------------------===//

using ModuleAnalysisManagerInstrumentPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::ModuleAnalysisManager,
                                    InstrumentPrototype>;

using InstrumentPrototypeAnalysisManagerModuleProxy =
    llvm::OuterAnalysisManagerProxy<InstrumentPrototypeAnalysisManager,
                                    llvm::Module>;

using FunctionAnalysisManagerInstrumentPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager,
                                    InstrumentPrototype>;

using MachineFunctionAnalysisManagerInstrumentPrototypeProxy =
    llvm::InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager,
                                    InstrumentPrototype>;

using InstrumentPrototypeAnalysisManagerFunctionProxy =
    llvm::OuterAnalysisManagerProxy<InstrumentPrototypeAnalysisManager,
                                    llvm::Function>;

using InstrumentPrototypeAnalysisManagerMachineFunctionProxy =
    llvm::OuterAnalysisManagerProxy<InstrumentPrototypeAnalysisManager,
                                    llvm::MachineFunction>;

/// \brief Adaptor that runs a single \c llvm::Module pass over the target
/// module of an \c InstrumentPrototype.
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

  llvm::PreservedAnalyses run(InstrumentPrototype &IP,
                              InstrumentPrototypeAnalysisManager &IPAM);

  void printPipeline(
      llvm::raw_ostream &OS,
      llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName);

  static bool isRequired() { return true; }

private:
  std::unique_ptr<PassConceptT> Pass;
};

/// \brief Adaptor that runs a single \c llvm::Module pass over the
/// instrumentation module of an \c InstrumentPrototype.
///
/// See \c RunOnTargetModuleAdaptor for the design rationale.
class RunOnInstrumentationModuleAdaptor
    : public llvm::PassInfoMixin<RunOnInstrumentationModuleAdaptor> {
public:
  using PassConceptT =
      llvm::detail::PassConcept<llvm::Module, llvm::ModuleAnalysisManager>;

  explicit RunOnInstrumentationModuleAdaptor(std::unique_ptr<PassConceptT> Pass)
      : Pass(std::move(Pass)) {}

  llvm::PreservedAnalyses run(InstrumentPrototype &IP,
                              InstrumentPrototypeAnalysisManager &IPAM);

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

void registerInstrumentPrototypeCrossLevelProxies(
    InstrumentPrototypeAnalysisManager &IP, llvm::ModuleAnalysisManager &MAM,
    llvm::FunctionAnalysisManager &FAM,
    llvm::MachineFunctionAnalysisManager &MFAM,
    llvm::PassInstrumentationCallbacks &PIC);

} // namespace luthier

//===----------------------------------------------------------------------===//
// Explicit specializations of the inner proxies' invalidation hooks.
//===----------------------------------------------------------------------===//
//
// These MUST live in \c namespace llvm: an explicit specialization of a member
// has to be declared in a namespace enclosing the specialized template.
namespace llvm {

template <>
bool luthier::ModuleAnalysisManagerInstrumentPrototypeProxy::Result::invalidate(
    luthier::InstrumentPrototype &IP, const PreservedAnalyses &PA,
    luthier::InstrumentPrototypeAnalysisManager::Invalidator &Inv);

template <>
bool luthier::FunctionAnalysisManagerInstrumentPrototypeProxy::Result::
    invalidate(luthier::InstrumentPrototype &IP, const PreservedAnalyses &PA,
               luthier::InstrumentPrototypeAnalysisManager::Invalidator &Inv);

template <>
bool luthier::MachineFunctionAnalysisManagerInstrumentPrototypeProxy::Result::
    invalidate(luthier::InstrumentPrototype &IP, const PreservedAnalyses &PA,
               luthier::InstrumentPrototypeAnalysisManager::Invalidator &Inv);

extern template class PassManager<luthier::InstrumentPrototype>;
extern template class AnalysisManager<luthier::InstrumentPrototype>;

} // namespace llvm

#endif
