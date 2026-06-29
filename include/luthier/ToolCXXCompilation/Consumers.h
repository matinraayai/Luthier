//===-- Consumers.h ----------------------------------------------*- C++-*-===//
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
/// Defines Sema and AST Consumers used in Luthier tool CXX compilation.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CXX_COMPILATION_CONSUMERS_H
#define LUTHIER_TOOL_CXX_COMPILATION_CONSUMERS_H
#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclGroup.h>
#include <clang/Sema/SemaConsumer.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>
#include <memory>

namespace clang {
class CompilerInstance;
class Decl;
class FunctionDecl;
class FunctionTemplateDecl;
class Sema;
} // namespace clang

namespace luthier {

/// The result of the pre-pass (see \c computeDevFuncExportPlan): the set of
/// \c __device__-only functions that have \b no \c __host__ overload anywhere
/// in the completed translation unit and therefore need one synthesized.
/// Entries are stable, ASTContext-independent location keys (the function's
/// canonical declaration location printed as \c file:line:col, or — for a
/// template specialization — its primary template's location).
struct DevFuncExportPlan {
  llvm::StringSet<> Synthesize;
};

/// Runs a throwaway, syntax-only pre-parse of the same translation unit and
/// returns which \c __device__-only functions lack a \c __host__ overload. The
/// real parse can only ask "does a host overload already exist?" with a
/// half-built AST (declaration order makes the answer unreliable — the standard
/// library declares the host \c malloc/\c sqrt after their \c __device__ peers),
/// whereas the pre-pass inspects the \e complete AST and gets it right. Returns
/// an empty plan for non-host CUDA/HIP compiles.
std::shared_ptr<const DevFuncExportPlan>
computeDevFuncExportPlan(clang::CompilerInstance &CI);

/// A \c clang::SemaConsumer that makes \c __device__ functions referenced from
/// host code (or carrying \c __attribute__((used))) host-addressable, so a
/// later IR pass can harvest a host-side handle for each of them.
///
/// Taking the address of a \c __device__ function from host code is normally
/// ill-formed (\c err_ref_bad_target), and by the time the AST is complete the
/// offending reference has already been rewritten into a \c RecoveryExpr — too
/// late to repair. The fix therefore has to happen \e during parsing: as each
/// top-level \c __device__-only function (or function template) is seen
/// (\c HandleTopLevelDecl), a body-less \c __host__ overload of it is
/// synthesized so that subsequent host references resolve against the host
/// overload instead of erroring. A body is deliberately \e not emitted yet, so
/// the synthesized declaration merges cleanly with any \c __host__ overload the
/// tool itself defines later in the same translation unit.
///
/// Once parsing is complete (\c HandleTranslationUnit) every host overload that
/// ended up referenced — or whose \c __device__ source was \c used — is
/// finalized: given an empty body if the tool did not define one, reconciled
/// with any user-written overload, and tagged with the export-handle
/// annotation the IR pass looks for.
class EmitHostHandleForDevFuncConsumer : public clang::SemaConsumer {
  clang::Sema *SemaRef{nullptr};

  /// A \c __device__-only function and the \c __host__ overload synthesized for
  /// it during parsing (the \c Host decl is body-less until finalization).
  struct SynthHandle {
    clang::FunctionDecl *Dev;
    clang::FunctionDecl *Host;
  };

  /// Synthesized non-template host overloads, in source order.
  llvm::SmallVector<SynthHandle, 16> Handles;

  /// Synthesized host overloads of \c __device__ function templates, keyed by
  /// the device template's canonical decl so each template is cloned once.
  llvm::DenseMap<clang::FunctionTemplateDecl *, clang::FunctionTemplateDecl *>
      TemplateHandles;

  /// \c __device__ functions that already had a host-callable counterpart (a
  /// user-written \c __host__ overload, or the function itself when it is
  /// \c __host__ \c __device__); their counterpart is annotated in place if the
  /// function turns out to be exported.
  llvm::SmallVector<clang::FunctionDecl *, 16> ExistingHosts;

  /// Pre-pass verdict: location keys of device functions that need a host
  /// overload synthesized (those lacking one in the complete AST).
  std::shared_ptr<const DevFuncExportPlan> Plan;

public:
  explicit EmitHostHandleForDevFuncConsumer(
      std::shared_ptr<const DevFuncExportPlan> Plan)
      : Plan(std::move(Plan)) {}

  void InitializeSema(clang::Sema &S) override;

  void ForgetSema() override;

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override;

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};

} // namespace luthier

#endif
