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
/// Defines the \c InstrumentationPassBuilder, a class for parsing and creating
/// instrumentation pipelines.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_BUILDER_H
#define LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_PASS_BUILDER_H

#include "luthier/ToolCodeGen/Prototype.h"
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
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

struct InstrumentationPMDriverOptions;

/// Facade around \c llvm::PassBuilder that owns the wrapped PB and hosts the
/// Luthier pipeline grammar over \c Prototype.
///
/// \pre Any \c llvm::PassInstrumentationCallbacks pointer passed to the
///      constructor must outlive this \c InstrumentationPassBuilder.
class InstrumentationPassBuilder {
public:
  /// Callback invoked while parsing a \c target(<name>) or
  /// \c instrumentation(<name>) block. Return \c true to signal the token was
  /// handled; \c false falls back to \c llvm::PassBuilder::parsePassPipeline.
  using ParseCallback =
      std::function<bool(llvm::StringRef /*InnerText*/,
                         PrototypePassManager & /*PPM*/, bool /*IsTarget*/)>;

  /// Fired once, right after the wrapped \c PassBuilder is constructed and
  /// Luthier passes are registered on it. Use it to install additional
  /// analyses, plugin passes, or pipeline-parsing callbacks that need to see
  /// the PB directly.
  using PassBuilderAugmentationCallback =
      std::function<void(llvm::PassBuilder &)>;

  /// Fired before adding any other passes to the IModule IR pipeline.
  using PreIROptimizationCallback =
      std::function<void(llvm::ModulePassManager &, llvm::OptimizationLevel)>;

  /// Fired before / after adding the Luthier IR intrinsic lowering pass to
  /// the IModule IR pipeline.
  using IntrinsicLoweringCallback =
      std::function<void(llvm::ModulePassManager &)>;

  /// Opaque handle for augmenting the codegen pipeline. The concrete
  /// implementation is \c .cpp-private.
  class CodeGenAugmenter;
  using AugmentCodeGenCallback =
      std::function<void(CodeGenAugmenter &, llvm::TargetMachine &)>;

  /// Primary constructor. Builds and owns an internal \c llvm::PassBuilder
  /// with the supplied arguments, then installs every Luthier pass and
  /// analysis on it via callbacks.
  InstrumentationPassBuilder(
      llvm::TargetMachine *TM, llvm::PipelineTuningOptions PTO = {},
                       std::optional<llvm::PGOOptions> PGOOpt = std::nullopt,
                       llvm::PassInstrumentationCallbacks *PIC = nullptr);

  ~InstrumentationPassBuilder();

  InstrumentationPassBuilder(const InstrumentationPassBuilder &) = delete;
  InstrumentationPassBuilder &
  operator=(const InstrumentationPassBuilder &) = delete;

  /// Returns the wrapped \c llvm::PassBuilder. Plugins register analyses,
  /// pipeline-parsing callbacks, and any pipeline-tuning options directly on
  /// it — but should prefer the extension-point registration helpers below
  /// when they exist.
  llvm::PassBuilder &getPassBuilder();

  /// Returns the target machine this builder was constructed with (may be
  /// null).
  llvm::TargetMachine *getTargetMachine() const { return TM; }

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
  void crossRegisterProxies(llvm::ModuleAnalysisManager &MAM,
                            llvm::FunctionAnalysisManager &FAM,
                            llvm::MachineFunctionAnalysisManager &MFAM,
                            PrototypeAnalysisManager &IPAM);

  /// Pre-register every zero-arg Luthier analysis on the appropriate manager,
  /// expanding \c LuthierPassRegistry.def. Passes/analyses that require
  /// constructor arguments must be registered by the caller.
  void registerAllAnalyses(llvm::ModuleAnalysisManager &MAM,
                           llvm::FunctionAnalysisManager &FAM,
                           llvm::MachineFunctionAnalysisManager &MFAM,
                           PrototypeAnalysisManager &IPAM);

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
      PrototypePassManager &PPM, const InstrumentationPMDriverOptions &Opts,
      llvm::OptimizationLevel Level, llvm::ModuleAnalysisManager &MAM,
      llvm::raw_pwrite_stream &Out, llvm::raw_pwrite_stream *DwoOut,
      llvm::CodeGenFileType FileType, llvm::MCContext &Ctx);

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

  /// Register a hook fired for every \c target(<inner>) /
  /// \c instrumentation(<inner>) block. Callbacks are tried in order; the
  /// first to return \c true owns the block.
  void registerParseCallback(ParseCallback Cb) {
    ParseCallbacks.push_back(std::move(Cb));
  }

  /// Register a callback fired once from the constructor, right after the
  /// wrapped PB is built and Luthier passes are installed on it.
  ///
  /// Note: because these fire in the constructor, callbacks registered
  /// *after* the constructor completes never run. This mirrors the
  /// equivalent slot in \c InstrumentationPMDriver, which took its
  /// augmentation callback as a constructor argument.
  void
  registerPassBuilderAugmentationCallback(PassBuilderAugmentationCallback Cb) {
    PassBuilderAugmentationCallbacks.push_back(std::move(Cb));
  }

  /// Register a callback fired at the very start of the IModule IR pipeline.
  void registerPreIROptimizationCallback(PreIROptimizationCallback Cb) {
    PreIROptimizationCallbacks.push_back(std::move(Cb));
  }

  /// Register a callback fired right before Luthier IR intrinsic lowering
  /// runs on the IModule.
  void registerPreIRIntrinsicLoweringCallback(IntrinsicLoweringCallback Cb) {
    PreIRIntrinsicLoweringCallbacks.push_back(std::move(Cb));
  }

  /// Register a callback fired right after Luthier IR intrinsic lowering
  /// runs on the IModule.
  void registerPostIRIntrinsicLoweringCallback(IntrinsicLoweringCallback Cb) {
    PostIRIntrinsicLoweringCallbacks.push_back(std::move(Cb));
  }

  /// Register a callback fired while the codegen pipeline is being assembled,
  /// after the internal AMDGPU codegen builder is constructed. Analogous to
  /// (and fixes the never-called) \c AugmentTargetPassConfigCallback slot in
  /// \c InstrumentationPMDriver.
  void registerAugmentCodeGenCallback(AugmentCodeGenCallback Cb) {
    AugmentCodeGenCallbacks.push_back(std::move(Cb));
  }

private:
  /// Install every Luthier pass and analysis on the wrapped PB via
  /// \c PassBuilder::registerPipelineParsingCallback and friends. Expands
  /// \c LuthierPassRegistry.def. Called once at construction.
  void registerLuthierPasses();

  /// Build the IModule IR pipeline (per-module default + intrinsic lowering).
  llvm::Error
  buildIROptimizationPipeline(llvm::ModulePassManager &IMPM,
                              const InstrumentationPMDriverOptions &Opts,
                              llvm::OptimizationLevel Level);

  /// Build the IModule codegen pipeline (ISel + machine passes with
  /// \c InjectedPayloadPEIPass splice). Delegates to
  /// \c GCNTargetMachine::buildCodeGenPipeline and then walks the resulting
  /// pass manager tree to insert Luthier's PEI pass after LLVM's stock one.
  llvm::Error buildCodeGenPipeline(llvm::ModulePassManager &IMPM,
                                   const InstrumentationPMDriverOptions &Opts,
                                   llvm::ModuleAnalysisManager &MAM,
                                   llvm::raw_pwrite_stream &Out,
                                   llvm::raw_pwrite_stream *DwoOut,
                                   llvm::CodeGenFileType FileType,
                                   llvm::MCContext &Ctx);

  llvm::TargetMachine *TM;
  llvm::PassInstrumentationCallbacks *PIC;
  std::unique_ptr<llvm::PassBuilder> PB;

  llvm::SmallVector<ParseCallback, 2> ParseCallbacks;
  llvm::SmallVector<PassBuilderAugmentationCallback, 2>
      PassBuilderAugmentationCallbacks;
  llvm::SmallVector<PreIROptimizationCallback, 2> PreIROptimizationCallbacks;
  llvm::SmallVector<IntrinsicLoweringCallback, 2>
      PreIRIntrinsicLoweringCallbacks;
  llvm::SmallVector<IntrinsicLoweringCallback, 2>
      PostIRIntrinsicLoweringCallbacks;
  llvm::SmallVector<AugmentCodeGenCallback, 2> AugmentCodeGenCallbacks;
};

} // namespace luthier

#endif
