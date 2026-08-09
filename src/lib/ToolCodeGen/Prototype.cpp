//===-- Prototype.cpp -------------------------------------------===//
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
/// Implements out-of-line definitions for \c Prototype.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/Prototype.h"
#include <cassert>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManagerImpl.h>

namespace luthier {

Prototype::Prototype(
    std::unique_ptr<llvm::Module> Target,
    std::unique_ptr<llvm::Module> IModule)
    : TargetModule(std::move(Target)), IModule(std::move(IModule)) {
  assert(this->TargetModule && this->IModule &&
         "Prototype modules must be non-null");
  assert(&this->TargetModule->getContext() == &this->IModule->getContext() &&
         "Prototype modules must share an LLVMContext");
}

void Prototype::forEachTargetMF(
    PrototypeAnalysisManager &PAM,
    llvm::function_ref<void(llvm::MachineFunction &)> Fn) {
  llvm::FunctionAnalysisManager &FAM =
      PAM.getResult<FunctionAnalysisManagerPrototypeProxy>(*this).getManager();
  for (llvm::Function &F : *TargetModule) {
    if (auto *MFRes = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F))
      Fn(MFRes->getMF());
  }
}

/// Runs \p Pass over \p M, which is the module of \p IP selected by the caller.
/// Mirrors LLVM's ModuleToFunctionPassAdaptor::run / the machinery in the other
/// LLVM Pass adaptors.
static llvm::PreservedAnalyses
runModulePass(RunOnTargetModuleAdaptor::PassConceptT &Pass, llvm::Module &M,
              Prototype &IP,
              PrototypeAnalysisManager &IPAM) {
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  // Request PassInstrumentation from the analysis manager; it drives the
  // instrumenting callbacks around the pass below.
  llvm::PassInstrumentation PI =
      IPAM.getResult<llvm::PassInstrumentationAnalysis>(IP);

  // Check the BeforePass callbacks; if asked to skip, do not run the pass and
  // report that everything is preserved.
  if (!PI.runBeforePass<llvm::Module>(Pass, M))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA = Pass.run(M, MAM);

  // The pass only touched module M, so directly reconcile the inner module
  // analysis manager here, invalidating whatever the pass did not preserve.
  MAM.invalidate(M, PA);

  PI.runAfterPass(Pass, M, PA);

  // We handled invalidation of module analyses above, so from the
  // Prototype pass manager's point of view all module analyses are
  // preserved. Keep the proxy live so the inner manager is not cleared.
  PA.preserveSet<llvm::AllAnalysesOn<llvm::Module>>();
  PA.preserve<ModuleAnalysisManagerPrototypeProxy>();
  return PA;
}

llvm::PreservedAnalyses
RunOnTargetModuleAdaptor::run(Prototype &IP,
                              PrototypeAnalysisManager &IPAM) {
  return runModulePass(*Pass, IP.getTargetModule(), IP, IPAM);
}

void RunOnTargetModuleAdaptor::printPipeline(
    llvm::raw_ostream &OS,
    llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName) {
  OS << "target(";
  Pass->printPipeline(OS, MapClassName);
  OS << ")";
}

llvm::PreservedAnalyses RunOnInstrumentationModuleAdaptor::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  return runModulePass(*Pass, IP.getInstrumentationModule(), IP, IPAM);
}

void RunOnInstrumentationModuleAdaptor::printPipeline(
    llvm::raw_ostream &OS,
    llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName) {
  OS << "instrumentation(";
  Pass->printPipeline(OS, MapClassName);
  OS << ")";
}

} // namespace luthier

// The invalidate specializations must be defined in namespace llvm (the
// namespace enclosing InnerAnalysisManagerProxy)/
namespace llvm {

using luthier::FunctionAnalysisManagerPrototypeProxy;
using luthier::MachineFunctionAnalysisManagerPrototypeProxy;
using luthier::ModuleAnalysisManagerPrototypeProxy;

template <>
bool ModuleAnalysisManagerPrototypeProxy::Result::invalidate(
    luthier::Prototype &IP, const llvm::PreservedAnalyses &PA,
    luthier::PrototypeAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<ModuleAnalysisManagerPrototypeProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<llvm::AllAnalysesOn<luthier::Prototype>>()) {
    InnerAM->clear(IP.getInstrumentationModule(),
                   IP.getInstrumentationModule().getName());
    InnerAM->clear(IP.getTargetModule(), IP.getTargetModule().getName());
    return true; // the proxy result itself is now invalid
  }
  return false;
}

template <>
bool FunctionAnalysisManagerPrototypeProxy::Result::invalidate(
    luthier::Prototype &IP, const llvm::PreservedAnalyses &PA,
    luthier::PrototypeAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<FunctionAnalysisManagerPrototypeProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<llvm::AllAnalysesOn<luthier::Prototype>>()) {
    InnerAM->clear();
    return true;
  }
  return false;
}

template <>
bool MachineFunctionAnalysisManagerPrototypeProxy::Result::invalidate(
    luthier::Prototype &IP, const llvm::PreservedAnalyses &PA,
    luthier::PrototypeAnalysisManager::Invalidator &Inv) {
  auto PAC =
      PA.getChecker<MachineFunctionAnalysisManagerPrototypeProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<llvm::AllAnalysesOn<luthier::Prototype>>()) {
    InnerAM->clear();
    return true;
  }
  return false;
}

} // namespace llvm

// Explicit template instantiation for IP
namespace llvm {
// PassManager::run emits a stack-trace entry that calls this per-IR-unit hook;
// LLVM only provides specializations for its own IR units (Module, Function),
// so IP must supply its own before the PassManager instantiation below.
template <>
void printIRUnitNameForStackTrace<luthier::Prototype>(
    raw_ostream &OS, const luthier::Prototype &IR) {
  OS << "prototype for \"" << IR.getName() << "\"";
}

template class PassManager<luthier::Prototype>;
template class AnalysisManager<luthier::Prototype>;
} // namespace llvm
