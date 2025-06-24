//===-- WrapperAnalysisPasses.h ---------------------------------*- C++ -*-===//
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
/// Describes a set of analysis passes that provide commonly used data
/// structures by the instrumentation passes in Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_WRAPPER_ANALYSIS_PASSES_H
#define LUTHIER_INSTRUMENTATION_WRAPPER_ANALYSIS_PASSES_H
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/PassManager.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {

/// \brief holds the mapping between the intrinsic inline assembly placeholder
/// indices and their \c IntrinsicIRLoweringInfo
class IntrinsicIRLoweringInfoMapAnalysis
    : public llvm::AnalysisInfoMixin<IntrinsicIRLoweringInfoMapAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IntrinsicIRLoweringInfoMapAnalysis>;

  static llvm::AnalysisKey Key;

public:
  class Result {
    std::vector<IntrinsicIRLoweringInfo> LoweringInfo{};
    Result() = default;
    friend class IntrinsicIRLoweringInfoMapAnalysis;

  public:
    [[nodiscard]] const std::vector<IntrinsicIRLoweringInfo> &
    getLoweringInfo() const {
      return LoweringInfo;
    }

    std::vector<IntrinsicIRLoweringInfo> &getLoweringInfo() {
      return LoweringInfo;
    }

    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  IntrinsicIRLoweringInfoMapAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) { return {}; }
};

/// \brief an analysis used by the instrumentation module passes that gives
/// access to the Target App's \c llvm::Module and
/// <tt>llvm::ModuleAnalysisManager</tt>, which in turn give access to
/// important analysis such as register liveness or state value storage
/// locations
class TargetAppModuleAndMAMAnalysis
    : public llvm::AnalysisInfoMixin<TargetAppModuleAndMAMAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<TargetAppModuleAndMAMAnalysis>;

  static llvm::AnalysisKey Key;

  llvm::ModuleAnalysisManager &TargetAppMAM;

  llvm::Module &TargetAppModule;

public:
  class Result {
    friend class TargetAppModuleAndMAMAnalysis;

    llvm::ModuleAnalysisManager &MAM;

    llvm::Module &M;

    Result(llvm::ModuleAnalysisManager &MAM, llvm::Module &M)
        : MAM(MAM), M(M) {};

  public:
    [[nodiscard]] llvm::ModuleAnalysisManager &getTargetAppMAM() const {
      return MAM;
    }

    [[nodiscard]] llvm::Module &getTargetAppModule() const { return M; }

    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit TargetAppModuleAndMAMAnalysis(
      llvm::ModuleAnalysisManager &TargetAppMAM, llvm::Module &TargetAppModule)
      : TargetAppMAM(TargetAppMAM), TargetAppModule(TargetAppModule) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return {TargetAppMAM, TargetAppModule};
  }
};

/// \brief An analysis which provides all the pass manager constructs
/// used for running IR passes and analysis on the instrumentation module
class IModulePMAnalysis : public llvm::AnalysisInfoMixin<IModulePMAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IModulePMAnalysis>;

  static llvm::AnalysisKey Key;

  llvm::Module &IModule;

  llvm::ModulePassManager &IPM;

  llvm::ModuleAnalysisManager &IMAM;

  llvm::LoopAnalysisManager &ILAM;

  llvm::FunctionAnalysisManager &IFAM;

  llvm::CGSCCAnalysisManager &ICGAM;

public:
  class Result {
    friend class IModulePMAnalysis;

    llvm::Module &IModule;
    llvm::ModulePassManager &IPM;
    llvm::ModuleAnalysisManager &IMAM;
    llvm::LoopAnalysisManager &ILAM;
    llvm::FunctionAnalysisManager &IFAM;
    llvm::CGSCCAnalysisManager &ICGAM;

    Result(llvm::Module &IModule, llvm::ModulePassManager &IPM,
           llvm::ModuleAnalysisManager &IMAM, llvm::LoopAnalysisManager &ILAM,
           llvm::FunctionAnalysisManager &IFAM,
           llvm::CGSCCAnalysisManager &ICGAM)
        : IModule(IModule), IPM(IPM), IMAM(IMAM), ILAM(ILAM), IFAM(IFAM),
          ICGAM(ICGAM) {};

  public:
    [[nodiscard]] llvm::Module &getModule() const { return IModule; }

    [[nodiscard]] llvm::ModulePassManager &getPM() const { return IPM; }

    [[nodiscard]] llvm::ModuleAnalysisManager &getMAM() const { return IMAM; }

    [[nodiscard]] llvm::LoopAnalysisManager &getLAM() const { return ILAM; }

    [[nodiscard]] llvm::FunctionAnalysisManager &getFAM() const { return IFAM; }

    [[nodiscard]] llvm::CGSCCAnalysisManager &getCGAM() const { return ICGAM; }

    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  IModulePMAnalysis(llvm::Module &IModule, llvm::ModulePassManager &IPM,
                    llvm::ModuleAnalysisManager &IMAM,
                    llvm::LoopAnalysisManager &ILAM,
                    llvm::FunctionAnalysisManager &IFAM,
                    llvm::CGSCCAnalysisManager &ICGAM)
      : IModule(IModule), IPM(IPM), IMAM(IMAM), ILAM(ILAM), IFAM(IFAM),
        ICGAM(ICGAM) {};

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    return {IModule, IPM, IMAM, ILAM, IFAM, ICGAM};
  }
};

/// \brief a legacy immutable pass that provides the instrumentation module's
/// \c llvm::ModueAnalysisManager to the legacy code gen passes
class IModuleMAMWrapperPass : public llvm::ImmutablePass {
private:
  llvm::ModuleAnalysisManager &IMAM;

public:
  static char ID;

  explicit IModuleMAMWrapperPass(llvm::ModuleAnalysisManager *IMAM = nullptr);

  [[nodiscard]] llvm::ModuleAnalysisManager &getMAM() const { return IMAM; }
};

} // namespace luthier

#endif