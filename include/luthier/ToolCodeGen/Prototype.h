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
#include <llvm/MC/MCRegister.h>
#include <variant>

namespace llvm {
class IRBuilderBase;
class PassInstrumentationCallbacks;
class Type;
class Value;
} // namespace llvm

namespace luthier {

class Prototype;

using PrototypeAnalysisManager = llvm::AnalysisManager<Prototype>;

using PrototypePassManager = llvm::PassManager<Prototype>;

/// A hook argument that names a physical register. The convenience
/// \c createInjectedPayload overload lowers each such entry to a
/// \c luthier::readReg intrinsic call whose result — of type \c Ty — is
/// forwarded to the hook in the argument's position.
struct RegArg {
  llvm::MCRegister Reg;
  llvm::Type *Ty;
};

/// Argument passed to the \c HookFn convenience overload of
/// \c Prototype::createInjectedPayload: either an already-built
/// \c llvm::Value* forwarded verbatim, or a \c RegArg that gets
/// materialized via a \c luthier::readReg intrinsic call in the payload
/// body before the hook is called.
using PayloadArg = std::variant<llvm::Value *, RegArg>;

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

  /// \brief Creates a new injected-payload function in the instrumentation
  /// module for \p TargetMI.
  ///
  /// \details A single entry \c BasicBlock is created and an \c IRBuilderBase
  /// pointing into it is passed to \p Build.
  ///
  /// \returns the newly created function, or an error if the operation fails
  llvm::Expected<llvm::Function *> createInjectedPayload(
      llvm::MachineInstr &TargetMI, llvm::FunctionAnalysisManager &IFAM,
      llvm::function_ref<llvm::Error(llvm::IRBuilderBase &)> Build);

  /// Convenience overload: creates an injected-payload function for \p TargetMI
  /// that calls \p HookFn with \p Args. \c Value* entries in \p Args are
  /// forwarded verbatim; \c RegArg entries are lowered to
  /// \c luthier::readReg intrinsic calls inside the payload and their
  /// results become the corresponding hook argument.
  llvm::Expected<llvm::Function *>
  createInjectedPayload(llvm::Function &HookFn, llvm::MachineInstr &TargetMI,
                        llvm::FunctionAnalysisManager &IFAM,
                        llvm::ArrayRef<PayloadArg> Args = {});

  /// \brief Invokes \p Fn on every <tt>llvm::MachineFunction</tt> of the
  /// target module.
  void forEachTargetMF(PrototypeAnalysisManager &PAM,
                       llvm::function_ref<void(llvm::MachineFunction &)> Fn);
};

//===----------------------------------------------------------------------===//
// Cross-level analysis-manager proxies
//===----------------------------------------------------------------------===//

/// Identifies which of a prototype's two modules an inner analysis manager
/// serves. Used only to give the proxies below distinct \c llvm::AnalysisKey s.
enum class PrototypeModuleKind { Target, Instrumentation };

/// \brief Prototype-level proxy for the analysis manager of type
/// \p AnalysisManagerT that serves the \p Kind module of the prototype.
///
/// \details Luthier's counterpart to \c llvm::InnerAnalysisManagerProxy. It
/// differs only in being keyed by \p Kind as well as by manager type, so that a
/// prototype can hand out one manager per module at every level below itself
/// instead of one manager shared by both.
///
/// The two modules must not share a manager. LLVM reaches a module's inner
/// managers through its own per-module proxies (\c
/// llvm::FunctionAnalysisManagerModuleProxy and friends), and their
/// invalidation hook responds to a module pass that does not preserve the proxy
/// by calling \c InnerAM->clear() — the whole manager, not the passing module's
/// share of it, and without consulting any cached result's own \c invalidate.
/// Fourteen of the module passes in the default O2 pipeline do exactly that
/// (\c AMDGPUAttributorPass, \c GlobalDCEPass and \c Annotation2MetadataPass
/// among them), as does the AMDGPU ISel pipeline. With a shared manager the
/// first of them to report a change destroys the target module's cached
/// \c llvm::MachineFunctionAnalysis results — and the <tt>MachineFunction</tt>s
/// those results own — even though the pass only ever ran on the
/// instrumentation module. Splitting the managers confines each clear to the
/// module whose pass triggered it.
template <typename AnalysisManagerT, PrototypeModuleKind Kind>
class PrototypeInnerAnalysisManagerProxy
    : public llvm::AnalysisInfoMixin<
          PrototypeInnerAnalysisManagerProxy<AnalysisManagerT, Kind>> {
public:
  class Result {
  public:
    explicit Result(AnalysisManagerT &InnerAM) : InnerAM(&InnerAM) {}

    /// The moved-from state is nulled out: this result carries the duty of
    /// clearing the manager, and only one copy of it may.
    Result(Result &&Arg) : InnerAM(Arg.InnerAM) { Arg.InnerAM = nullptr; }

    Result &operator=(Result &&RHS) {
      InnerAM = RHS.InnerAM;
      RHS.InnerAM = nullptr;
      return *this;
    }

    /// Clears the manager if this result is destroyed without having seen an
    /// invalidate call, mirroring \c llvm::InnerAnalysisManagerProxy::Result.
    ~Result() {
      if (InnerAM)
        InnerAM->clear();
    }

    AnalysisManagerT &getManager() { return *InnerAM; }

    bool invalidate(Prototype &IP, const llvm::PreservedAnalyses &PA,
                    typename PrototypeAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<PrototypeInnerAnalysisManagerProxy>();
      if (!PAC.preserved() &&
          !PAC.template preservedSet<llvm::AllAnalysesOn<Prototype>>()) {
        InnerAM->clear();
        return true; // the proxy result itself is now invalid
      }
      return false;
    }

  private:
    AnalysisManagerT *InnerAM;
  };

  explicit PrototypeInnerAnalysisManagerProxy(AnalysisManagerT &InnerAM)
      : InnerAM(&InnerAM) {}

  Result run(Prototype &IP, PrototypeAnalysisManager &AM) {
    return Result(*InnerAM);
  }

private:
  friend llvm::AnalysisInfoMixin<
      PrototypeInnerAnalysisManagerProxy<AnalysisManagerT, Kind>>;

  static llvm::AnalysisKey Key;

  AnalysisManagerT *InnerAM;
};

template <typename AnalysisManagerT, PrototypeModuleKind Kind>
llvm::AnalysisKey
    PrototypeInnerAnalysisManagerProxy<AnalysisManagerT, Kind>::Key;

using TargetModuleAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::ModuleAnalysisManager,
                                       PrototypeModuleKind::Target>;

using IModuleAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::ModuleAnalysisManager,
                                       PrototypeModuleKind::Instrumentation>;

using TargetFunctionAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::FunctionAnalysisManager,
                                       PrototypeModuleKind::Target>;

using IModuleFunctionAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::FunctionAnalysisManager,
                                       PrototypeModuleKind::Instrumentation>;

using TargetMachineFunctionAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager,
                                       PrototypeModuleKind::Target>;

using IModuleMachineFunctionAnalysisManagerPrototypeProxy =
    PrototypeInnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager,
                                       PrototypeModuleKind::Instrumentation>;

/// The outer proxies are one type each: they are registered separately on each
/// module's managers, and both registrations point back at the same
/// \c PrototypeAnalysisManager.
using PrototypeAnalysisManagerModuleProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::Module>;

using PrototypeAnalysisManagerFunctionProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::Function>;

using PrototypeAnalysisManagerMachineFunctionProxy =
    llvm::OuterAnalysisManagerProxy<PrototypeAnalysisManager,
                                    llvm::MachineFunction>;

/// \note Any Prototype-level pass reporting less than
/// \c PreservedAnalyses::all() must state explicitly which of the six inner
/// analysis-manager proxies it preserves. A proxy it does not name is dropped,
/// and a dropped proxy clears the manager behind it — throwing away that
/// module's cached \c llvm::MachineFunctionAnalysis results along with the
/// <tt>MachineFunction</tt>s those results own. Naming only the proxies of the
/// module a pass actually disturbed now leaves the other module untouched,
/// which is the whole point of splitting them.

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

namespace llvm {

extern template class PassManager<luthier::Prototype>;
extern template class AnalysisManager<luthier::Prototype>;

} // namespace llvm

#endif
