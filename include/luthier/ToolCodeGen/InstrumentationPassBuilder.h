//===-- InstrumentationPassBuilder.h ----------------------------*- C++ -*-===//
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
/// Defines the \c InstrumentationPassBuilder, a class for parsing, creating,
/// and augmenting the standard instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_BUILDER_H
#define LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_BUILDER_H

#include "EntryPoint.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <optional>

namespace llvm {
class GCNTargetMachine;
struct CGPassBuilderOption;
class TargetMachine;
class PassInstrumentationCallbacks;
class MCContext;
class raw_pwrite_stream;
struct PGOOptions;
class PipelineTuningOptions;
} // namespace llvm

namespace luthier {

/// Primary interface used to construct an instrumentation pipeline.
class InstrumentationPassBuilder {
public:
  /// Callback invoked while parsing a \c target(<name>) or
  /// \c instrumentation(<name>) block. Return \c true to signal the token was
  /// handled; \c false falls back to \c llvm::PassBuilder::parsePassPipeline.
  using ParseCallback =
      std::function<bool(llvm::StringRef /*InnerText*/,
                         PrototypePassManager & /*PPM*/, bool /*IsTarget*/)>;

  /// Callback for before any code is discovered from the initial entry point.
  using PreCodeDiscoveryCallback =
      std::function<void(PrototypePassManager &, llvm::OptimizationLevel)>;

  /// Callback for before any instrumentation passes are run on the discovered
  /// code.
  using PreInstrumentationCallback =
      std::function<void(PrototypePassManager &, llvm::OptimizationLevel)>;

  /// Callback for before running IR optimization passes on the instrumentation
  /// module.
  using PreInstrumentationOptimizationCallback =
      std::function<void(PrototypePassManager &, llvm::OptimizationLevel)>;

  /// Callback for before ISEL is performed on the instrumentation module.
  using PreInstrumentationISelCallback =
      std::function<void(PrototypePassManager &, llvm::OptimizationLevel)>;

  /// Callback for before codegen passes are applied on the instrumentation
  /// module.
  using PreInstrumentationCodeGenPassesCallback =
      std::function<void(PrototypePassManager &, llvm::OptimizationLevel)>;

  /// Primary constructor. Builds and owns an internal \c llvm::PassBuilder
  /// with the supplied arguments, then installs every Luthier pass and
  /// analysis on it via callbacks.
  InstrumentationPassBuilder(
      llvm::TargetMachine &TM, llvm::PipelineTuningOptions PTO = {},
      std::optional<llvm::PGOOptions> PGOOpt = std::nullopt,
      llvm::PassInstrumentationCallbacks *PIC = nullptr);

  ~InstrumentationPassBuilder();

  InstrumentationPassBuilder(const InstrumentationPassBuilder &) = delete;
  InstrumentationPassBuilder &
  operator=(const InstrumentationPassBuilder &) = delete;

  /// Returns the target machine this builder was constructed with
  llvm::TargetMachine &getTargetMachine() const { return TM; }

  /// Returns the \c PassInstrumentationCallbacks pointer this builder was
  /// constructed with (may be null).
  llvm::PassInstrumentationCallbacks *getPassInstrumentationCallbacks() const {
    return PIC;
  }

  /// \brief The LLVM analysis managers serving a single module of a
  /// \c Prototype.
  ///
  /// \details A prototype's two modules get one bundle each. They may not be
  /// shared: LLVM's per-module proxies clear the inner manager they reach
  /// whenever a module pass does not preserve them, so one shared bundle lets a
  /// pass over one module destroy the other's results. See
  /// \c luthier::PrototypeInnerAnalysisManagerProxy.
  struct ModuleAnalysisManagers {
    llvm::ModuleAnalysisManager &MAM;
    llvm::CGSCCAnalysisManager &CGAM;
    llvm::FunctionAnalysisManager &FAM;
    llvm::LoopAnalysisManager &LAM;
    llvm::MachineFunctionAnalysisManager &MFAM;
  };

  /// Register the cross-level analysis-manager proxies that let passes running
  /// over an \c Prototype reach the per-module, per-function, and
  /// per-machine-function analyses of either of its modules (and vice-versa).
  /// Also registers \c PassInstrumentationAnalysis on \p PAM using the PIC held
  /// by the wrapped \c llvm::PassBuilder, and wires up the inner levels of both
  /// \p Target and \p Instrumentation via
  /// \c llvm::PassBuilder::crossRegisterProxies.
  ///
  /// Modeled on \c llvm::PassBuilder::crossRegisterProxies; call this once.
  void crossRegisterProxies(PrototypeAnalysisManager &PAM,
                            const ModuleAnalysisManagers &Target,
                            const ModuleAnalysisManagers &Instrumentation);

  void registerPrototypeAnalyses(PrototypeAnalysisManager &PAM);

  void registerModuleAnalyses(llvm::ModuleAnalysisManager &MAM);

  void registerCGSCCAnalyses(llvm::CGSCCAnalysisManager &CGAM);

  void registerFunctionAnalyses(llvm::FunctionAnalysisManager &FAM);

  void registerLoopAnalyses(llvm::LoopAnalysisManager &LAM);

  void
  registerMachineFunctionAnalyses(llvm::MachineFunctionAnalysisManager &MFAM);

  /// Convenience wrapper registering every level's analyses on \p AMs; call
  /// once per module of the prototype.
  void registerAnalyses(const ModuleAnalysisManagers &AMs);

  /// Parse a top-level pipeline string of the form
  ///   target(<inner>) [, instrumentation(<inner>) ...]
  /// and add the resulting adaptors to \p PPM. Bare pass names at the top
  /// level are an error. Each inner text is parsed via
  /// \c llvm::PassBuilder::parsePassPipeline unless a registered
  /// \c ParseCallback handled it first.
  llvm::Error parsePipeline(PrototypePassManager &PPM,
                            llvm::StringRef PipelineText);

  /// \brief Assemble the standard instrumentation pipeline into \p PPM.
  ///
  /// \details Runs code discovery, hands \p InstCallback the chance to add the
  /// passes that create injected payloads, optimizes and lowers the
  /// instrumentation module, drives AMDGPU codegen over it, and finally patches
  /// the result back into the target module. Every registered
  /// \c register*Callback hook is invoked at its corresponding phase. When
  /// \p Out is non-null an asm printer is appended for the target module.
  ///
  /// The target machine is the one this builder was constructed with; codegen
  /// needs it as a \c GCNTargetMachine, and narrowing it here rather than in the
  /// signature keeps AMDGPU-internal types out of this header.
  llvm::Error buildInstrumentationPipeline(
      PrototypePassManager &PPM, PreInstrumentationCallback InstCallback,
      llvm::OptimizationLevel Level, llvm::CodeGenFileType FileType,
      llvm::CGPassBuilderOption &CGPBO, llvm::raw_pwrite_stream *Out,
      llvm::PassInstrumentationCallbacks *PIC);

  /// Wrap \p Pass so it runs over the target module of the prototype and
  /// append it to \p PPM.
  template <typename ModulePassT>
  void addTargetModulePass(PrototypePassManager &PPM, ModulePassT &&Pass) {
    PPM.addPass(
        createRunOnTargetModuleAdaptor(std::forward<ModulePassT>(Pass)));
  }

  /// Wrap \p Pass so it runs over the instrumentation module of the
  /// prototype and append it to \p PPM.
  template <typename ModulePassT>
  void addInstrumentationModulePass(PrototypePassManager &PPM,
                                    ModulePassT &&Pass) {
    PPM.addPass(createRunOnInstrumentationModuleAdaptor(
        std::forward<ModulePassT>(Pass)));
  }

  void registerPipelineParsingCallback(ParseCallback Cb) {
    ParseCallbacks.push_back(std::move(Cb));
  }

  void registerPipelineParsingCallback(
      const std::function<
          bool(llvm::StringRef Name, llvm::CGSCCPassManager &,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>)> &C) {
    PB->registerPipelineParsingCallback(C);
  }

  void registerPipelineParsingCallback(
      const std::function<
          bool(llvm::StringRef Name, llvm::FunctionPassManager &,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>)> &C) {
    PB->registerPipelineParsingCallback(C);
  }

  void registerPipelineParsingCallback(
      const std::function<
          bool(llvm::StringRef Name, llvm::LoopPassManager &,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>)> &C) {
    PB->registerPipelineParsingCallback(C);
  }

  void registerPipelineParsingCallback(
      const std::function<
          bool(llvm::StringRef Name, llvm::ModulePassManager &,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>)> &C) {
    PB->registerPipelineParsingCallback(C);
  }

  void registerPipelineParsingCallback(
      const std::function<
          bool(llvm::StringRef Name, llvm::MachineFunctionPassManager &,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>)> &C) {
    PB->registerPipelineParsingCallback(C);
  }

  void registerPreCodeDiscoveryCallback(PreCodeDiscoveryCallback CB) {
    PreCodeDiscoveryCallBacks.push_back(CB);
  }

  void registerPreInstrumentationCallback(PreInstrumentationCallback CB) {
    PreInstrumentationCallbacks.push_back(CB);
  }

  void registerPreInstrumentationOptimizationCallback(
      PreInstrumentationOptimizationCallback CB) {
    PreInstrumentationOptimizationCallbacks.push_back(CB);
  }

  void
  registerPreInstrumentationISelCallback(PreInstrumentationISelCallback CB) {
    PreInstrumentationISelCallbacks.push_back(CB);
  }

  void registerPreInstrumentationCodeGenPassesCallback(
      PreInstrumentationCodeGenPassesCallback CB) {
    PreInstrumentationCodeGenPassesCallbacks.push_back(CB);
  }

private:

  llvm::TargetMachine &TM;
  llvm::PassInstrumentationCallbacks *PIC;
  std::unique_ptr<llvm::PassBuilder> PB;

  /// \brief Instrumentation callbacks used for the \c Prototype level of the
  /// pipeline, deliberately kept empty.
  ///
  /// \details LLVM's \c StandardInstrumentations dispatches on the IR unit type
  /// via \c llvm::Any and \c llvm_unreachable s on anything outside
  /// Module/Function/SCC/Loop/MachineFunction (see \c getIRName in
  /// \c StandardInstrumentations.cpp). \c Prototype is a Luthier-only IR unit,
  /// so routing \c PassManager<Prototype> 's before/after-pass callbacks
  /// through the PIC that \c StandardInstrumentations registered against would
  /// abort — or, in an NDEBUG build, segfault — on the very first pass. The
  /// nested Module/Function/MachineFunction levels keep using the real PIC, so
  /// \c -print-after-all and friends still work for everything LLVM can name.
  llvm::PassInstrumentationCallbacks PrototypePIC;

  llvm::SmallVector<ParseCallback, 2> ParseCallbacks;
  llvm::SmallVector<PreCodeDiscoveryCallback, 2> PreCodeDiscoveryCallBacks;
  llvm::SmallVector<PreInstrumentationCallback, 2> PreInstrumentationCallbacks;
  llvm::SmallVector<PreInstrumentationOptimizationCallback, 2>
      PreInstrumentationOptimizationCallbacks;
  llvm::SmallVector<PreInstrumentationISelCallback, 2>
      PreInstrumentationISelCallbacks;
  llvm::SmallVector<PreInstrumentationCodeGenPassesCallback, 2>
      PreInstrumentationCodeGenPassesCallbacks;
};

} // namespace luthier

#endif
