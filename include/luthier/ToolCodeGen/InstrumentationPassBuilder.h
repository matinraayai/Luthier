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
      std::function<void(llvm::ModulePassManager &, llvm::OptimizationLevel)>;

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

  /// Register the cross-level analysis-manager proxies that let passes
  /// running over an \c Prototype reach the per-module, per-function, and
  /// per-machine-function analyses on \p MAM, \p FAM, and \p MFAM (and
  /// vice-versa). Also registers \c PassInstrumentationAnalysis on \p IPAM
  /// using the PIC held by the wrapped \c llvm::PassBuilder.
  ///
  /// Modeled on \c llvm::PassBuilder::crossRegisterProxies; call this once
  /// after \c PB.crossRegisterProxies has wired up the inner levels.
  void crossRegisterProxies(PrototypeAnalysisManager &PAM,
                            llvm::ModuleAnalysisManager &MAM,
                            llvm::CGSCCAnalysisManager &CGAM,
                            llvm::FunctionAnalysisManager &FAM,
                            llvm::LoopAnalysisManager &LAM,
                            llvm::MachineFunctionAnalysisManager &MFAM);

  void registerPrototypeAnalyses(PrototypeAnalysisManager &PAM);

  void registerModuleAnalyses(llvm::ModuleAnalysisManager &MAM);

  void registerCGSCCAnalyses(llvm::CGSCCAnalysisManager &CGAM);

  void registerFunctionAnalyses(llvm::FunctionAnalysisManager &FAM);

  void registerLoopAnalyses(llvm::LoopAnalysisManager &LAM);

  void
  registerMachineFunctionAnalyses(llvm::MachineFunctionAnalysisManager &MFAM);

  /// Parse a top-level pipeline string of the form
  ///   target(<inner>) [, instrumentation(<inner>) ...]
  /// and add the resulting adaptors to \p PPM. Bare pass names at the top
  /// level are an error. Each inner text is parsed via
  /// \c llvm::PassBuilder::parsePassPipeline unless a registered
  /// \c ParseCallback handled it first.
  llvm::Error parsePipeline(PrototypePassManager &PPM,
                            llvm::StringRef PipelineText);

  /// Construct the full Luthier end-to-end instrumentation pipeline on \p PPM,
  /// mirroring the behavior of \c InstrumentationPMDriver::run:
  ///
  ///   1. IModule IR pipeline
  ///        - Pre-IR-optimization callbacks
  ///        - buildPerModuleDefaultPipeline(Level)
  ///        - Pre-intrinsic-lowering callbacks
  ///        - RequireAnalysis<InjectedPayloadSideEffectsAnalysis>
  ///        - ProcessIntrinsicsAtIRLevelPass
  ///        - Post-intrinsic-lowering callbacks
  ///   2. IModule codegen pipeline built by
  ///        \c GCNTargetMachine::buildCodeGenPipeline with
  ///        \c InjectedPayloadPEIPass spliced immediately after LLVM's
  ///        \c PrologEpilogInserterPass, unless
  ///        \c Opts.DisableInjectedPayloadPEI is set.
  ///   3. Prototype-level passes:
  ///        \c IntrinsicMIRLoweringPass,
  ///        \c IModuleIPPredicatedLivenessAnalysis (require),
  ///        \c InjectedPayloadPreserveLiveRegsPass,
  ///        \c SVAPhysVGPRPinPass.
  ///   4. \c TargetModulePatcherPass.
  ///
  /// \p MAM, \p Out, \p DwoOut, \p FileType, and \p Ctx are forwarded to
  /// \c GCNTargetMachine::buildCodeGenPipeline. Callers append their own
  /// \c NewPMAsmPrinter afterwards (or use \p Out for AsmPrinter output
  /// depending on \p FileType).
  llvm::Error buildInstrumentationPipeline(
      PrototypePassManager &PPM, llvm::OptimizationLevel Level,
      llvm::raw_pwrite_stream &Out, llvm::raw_pwrite_stream *DwoOut,
      llvm::CodeGenFileType FileType, llvm::MCContext &Ctx,
      const llvm::amdhsa::kernel_descriptor_t &InitialExecutionPoint,
      EntryPoint InitialEntryPoint);

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

  void registerParseCallback(ParseCallback Cb) {
    ParseCallbacks.push_back(std::move(Cb));
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
  /// Build the IModule IR pipeline (per-module default + intrinsic lowering).
  llvm::Error buildIROptimizationPipeline(llvm::ModulePassManager &IMPM,
                                          llvm::OptimizationLevel Level);

  /// Build the IModule codegen pipeline (ISel + machine passes with
  /// \c InjectedPayloadPEIPass splice). Delegates to
  /// \c GCNTargetMachine::buildCodeGenPipeline and then walks the resulting
  /// pass manager tree to insert Luthier's PEI pass after LLVM's stock one.
  llvm::Error buildCodeGenPipeline(llvm::ModulePassManager &IMPM,
                                   llvm::ModuleAnalysisManager &MAM,
                                   llvm::raw_pwrite_stream &Out,
                                   llvm::raw_pwrite_stream *DwoOut,
                                   llvm::CodeGenFileType FileType,
                                   llvm::MCContext &Ctx);

  llvm::TargetMachine &TM;
  llvm::PassInstrumentationCallbacks *PIC;
  std::unique_ptr<llvm::PassBuilder> PB;

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
