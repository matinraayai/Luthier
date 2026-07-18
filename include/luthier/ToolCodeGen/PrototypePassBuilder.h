//===-- PrototypePassBuilder.h ------------------------*- C++ -*-===//
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
/// Defines \c PrototypePassBuilder, a thin wrapper around
/// \c llvm::PassBuilder that lets callers assemble pass pipelines over the
/// \c Prototype IR unit.  The wrapper owns the top-level pipeline
/// grammar (\c target(...) / \c instrumentation(...)) and provides typed
/// helpers for adding module passes to either side of the prototype.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PROTOTYPE_PASS_BUILDER_H
#define LUTHIER_TOOL_CODE_GEN_PROTOTYPE_PASS_BUILDER_H

#include "luthier/ToolCodeGen/Prototype.h"
#include <functional>
#include <llvm/ADT/StringRef.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <vector>

namespace luthier {

/// Thin wrapper around \c llvm::PassBuilder that lets callers (including
/// plugins) add passes to an \c PrototypePassManager.  Plugins
/// should treat \c getPassBuilder() as the standard extension surface for
/// analyses and named-pass registration; the wrapper-level helpers are for
/// composing passes over the Prototype IR unit.
class PrototypePassBuilder {
public:
  /// Callback invoked while parsing a \c target(<name>) or
  /// \c instrumentation(<name>) block.  The callback is given the inner
  /// text, the pass manager it should add to, and a flag identifying which
  /// side of the prototype the block belongs to.  Return \c true to signal
  /// the token was handled; \c false falls back to
  /// \c llvm::PassBuilder::parsePassPipeline.
  using ParseCallback = std::function<bool(
      llvm::StringRef /*InnerText*/,
      PrototypePassManager & /*IPPM*/, bool /*IsTarget*/)>;

  explicit PrototypePassBuilder(llvm::PassBuilder &PB) : PB(PB) {}

  /// Returns the wrapped \c llvm::PassBuilder.  Plugins register analyses,
  /// pipeline-parsing callbacks, and any pipeline-tuning options directly
  /// against it.
  llvm::PassBuilder &getPassBuilder() { return PB; }

  /// Register the cross-level analysis-manager proxies that let passes
  /// running over an \c Prototype reach the per-module,
  /// per-function, and per-machine-function analyses on \p MAM, \p FAM,
  /// and \p MFAM (and vice-versa).  Also registers
  /// \c PassInstrumentationAnalysis on \p IPAM using the PIC held by the
  /// wrapped \c llvm::PassBuilder.
  ///
  /// Modeled on \c llvm::PassBuilder::crossRegisterProxies; call this once
  /// after \c PB.crossRegisterProxies has wired up the inner levels.
  void crossRegisterProxies(llvm::ModuleAnalysisManager &MAM,
                            llvm::FunctionAnalysisManager &FAM,
                            llvm::MachineFunctionAnalysisManager &MFAM,
                            PrototypeAnalysisManager &IPAM);

  /// Parse a top-level pipeline string of the form
  ///   target(<inner>) [, instrumentation(<inner>) ...]
  /// and add the resulting adaptors to \p IPPM.  Bare pass names at the
  /// top level are an error.  Each inner text is parsed via
  /// \c llvm::PassBuilder::parsePassPipeline unless a registered
  /// \c ParseCallback handled it first.
  llvm::Error parsePipeline(PrototypePassManager &IPPM,
                            llvm::StringRef PipelineText);

  /// Wrap \p Pass so it runs over the target module of the prototype and
  /// append it to \p IPPM.
  template <typename ModulePassT>
  void addTargetModulePass(PrototypePassManager &IPPM,
                           ModulePassT &&Pass) {
    IPPM.addPass(createRunOnTargetModuleAdaptor(
        std::forward<ModulePassT>(Pass)));
  }

  /// Wrap \p Pass so it runs over the instrumentation module of the
  /// prototype and append it to \p IPPM.
  template <typename ModulePassT>
  void addInstrumentationModulePass(PrototypePassManager &IPPM,
                                    ModulePassT &&Pass) {
    IPPM.addPass(createRunOnInstrumentationModuleAdaptor(
        std::forward<ModulePassT>(Pass)));
  }

  /// Register a hook fired for every \c target(<inner>) /
  /// \c instrumentation(<inner>) block.  All registered callbacks are
  /// tried in order; the first to return \c true owns the block.
  void registerParseCallback(ParseCallback Cb) {
    ParseCallbacks.push_back(std::move(Cb));
  }

private:
  llvm::PassBuilder &PB;
  std::vector<ParseCallback> ParseCallbacks;
};

} // namespace luthier

#endif
