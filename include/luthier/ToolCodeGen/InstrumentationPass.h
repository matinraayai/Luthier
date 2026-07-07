//===-- InstrumentationPass.h -------------------------------------*-C++-*-===//
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
/// Defines the \c InstrumentationPass CRTP base class for passes used in the
/// instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_H
#define LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_H
#include "luthier/ToolCodeGen/WrapperAnalysisPasses.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Preserved-analyses pair returned by
/// \c InstrumentationPass::runInstrumentationPass
struct InstrumentationPreservedAnalyses {
  /// Describes which IModule analyses are still valid after the call
  llvm::PreservedAnalyses IModulePA = llvm::PreservedAnalyses::all();
  /// Describes which target-module analyses are still valid
  llvm::PreservedAnalyses TargetPA = llvm::PreservedAnalyses::all();
};

/// \brief Primary CRTP template for instrumentation pass pipeline
/// \details Two template parameters independently select the iteration
/// granularity on the IModule side (<tt>IModuleUnitT</tt>) and the
/// target-module side (<tt>TargetUnitT</tt>).  The four supported combinations/// are:
///
/// | IModuleUnitT | TargetUnitT     | runInstrumentationPass() signature
/// |--------------|-----------------|------------------------------------------
/// | Module       | Module          | (IModule, IMAM, TargetModule, TargetMAM)
/// | Function     | Module          | (IFunc,   IFAM, TargetModule, TargetMAM)
/// | Module       | Function        | (IModule, IMAM, TargetFunc,   TargetFAM)
/// | Module       | MachineFunction | (IModule, IMAM, TargetMF,     TargetFAM)
///
/// In every case the IModule-side arguments come first, followed by the
/// target-side arguments.
///
/// \c runInstrumentationPass returns an \c InstrumentationPreservedAnalyses
/// carrying one \c PreservedAnalyses for each side.  The base class uses the
/// IModule PA to drive IModule-side analysis invalidation (passed back to t
/// he pass manager via \c run's return value) and the target PA to explicitly
/// invalidate the target-side analysis manager, which lives outside the normal
/// PM flow
template <typename Derived, typename IModuleUnitT = llvm::Module,
          typename TargetUnitT = llvm::Module>
class InstrumentationPass;

// ---------------------------------------------------------------------------
// <Module, Module>
// ---------------------------------------------------------------------------

template <typename Derived>
class InstrumentationPass<Derived, llvm::Module, llvm::Module>
    : public llvm::PassInfoMixin<Derived> {
public:
  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &IMAM) {
    auto &TR = IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
    llvm::Module &TargetModule = TR.getTargetAppModule();
    llvm::ModuleAnalysisManager &TargetMAM = TR.getTargetAppMAM();

    auto IPA = static_cast<Derived *>(this)->runInstrumentationPass(
        IModule, IMAM, TargetModule, TargetMAM);
    TargetMAM.invalidate(TargetModule, IPA.TargetPA);
    return IPA.IModulePA;
  }
};

// ---------------------------------------------------------------------------
// <Function, Module>
// ---------------------------------------------------------------------------

template <typename Derived>
class InstrumentationPass<Derived, llvm::Function, llvm::Module>
    : public llvm::PassInfoMixin<Derived> {
public:
  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &IMAM) {
    auto &TR = IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
    llvm::Module &TargetModule = TR.getTargetAppModule();
    llvm::ModuleAnalysisManager &TargetMAM = TR.getTargetAppMAM();
    auto &IFAM =
        IMAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
            .getManager();

    llvm::PreservedAnalyses IModulePA = llvm::PreservedAnalyses::all();
    llvm::PreservedAnalyses TargetPA = llvm::PreservedAnalyses::all();
    for (llvm::Function &F : IModule) {
      if (F.isDeclaration())
        continue;
      auto IPA = static_cast<Derived *>(this)->runInstrumentationPass(
          F, IFAM, TargetModule, TargetMAM);
      IFAM.invalidate(F, IPA.IModulePA);
      IModulePA.intersect(IPA.IModulePA);
      TargetPA.intersect(IPA.TargetPA);
    }
    TargetMAM.invalidate(TargetModule, TargetPA);
    return IModulePA;
  }
};

// ---------------------------------------------------------------------------
// <Module, Function>
// ---------------------------------------------------------------------------

template <typename Derived>
class InstrumentationPass<Derived, llvm::Module, llvm::Function>
    : public llvm::PassInfoMixin<Derived> {
public:
  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &IMAM) {
    auto &TR = IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
    llvm::Module &TargetModule = TR.getTargetAppModule();
    llvm::ModuleAnalysisManager &TargetMAM = TR.getTargetAppMAM();
    auto &TargetFAM =
        TargetMAM
            .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
            .getManager();

    llvm::PreservedAnalyses IModulePA = llvm::PreservedAnalyses::all();
    for (llvm::Function &F : TargetModule) {
      if (F.isDeclaration())
        continue;
      auto IPA = static_cast<Derived *>(this)->runInstrumentationPass(
          IModule, IMAM, F, TargetFAM);
      TargetFAM.invalidate(F, IPA.TargetPA);
      IModulePA.intersect(IPA.IModulePA);
    }
    return IModulePA;
  }
};

// ---------------------------------------------------------------------------
// <Module, MachineFunction>
// ---------------------------------------------------------------------------

template <typename Derived>
class InstrumentationPass<Derived, llvm::Module, llvm::MachineFunction>
    : public llvm::PassInfoMixin<Derived> {
public:
  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &IMAM) {
    auto &TR = IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
    llvm::Module &TargetModule = TR.getTargetAppModule();
    llvm::ModuleAnalysisManager &TargetMAM = TR.getTargetAppMAM();
    auto &TargetFAM =
        TargetMAM
            .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
            .getManager();

    llvm::PreservedAnalyses IModulePA = llvm::PreservedAnalyses::all();
    for (llvm::Function &F : TargetModule) {
      auto *MFRes = TargetFAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
      if (!MFRes)
        continue;
      auto IPA = static_cast<Derived *>(this)->runInstrumentationPass(
          IModule, IMAM, MFRes->getMF(), TargetFAM);
      TargetFAM.invalidate(F, IPA.TargetPA);
      IModulePA.intersect(IPA.IModulePA);
    }
    return IModulePA;
  }
};

} // namespace luthier

#endif
