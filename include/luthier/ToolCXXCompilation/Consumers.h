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
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>

namespace clang {
class CompilerInstance;
class Decl;
class FunctionDecl;
class FunctionTemplateDecl;
class Sema;
} // namespace clang

namespace luthier {

/// A \c clang::SemaConsumer that makes \c __device__ functions referenced from
/// host code (or carrying <tt>__attribute__((used))</tt>) host-addressable by
/// emitting an identical \c __host__ function with an empty body for them.
/// If an associated \c __host__ function already exists, or if the
/// \c __device__ function itself is also \c __host__, then they are annotated
/// to be the corresponding host handle. The synthesized host handles'
/// prototypes will be identical to their device sibling in every way, only
/// removing anything device-specific from them.
///
/// \details Taking the address of a \c __device__ function from host code is
/// normally ill-formed (\c clang::err_ref_bad_target), and by the time the AST
/// is complete the offending reference has already been rewritten into a
/// \c RecoveryExpr - too late to repair. Therefore, this consumer does its job
/// in two steps:
/// - Step 1: Perform a syntax-only front-end action before the main parse
/// happens (i.e., on consumer construction) to figure out in advance which
/// \c __device__ functions or templates require emitting a \c __host__ handle.
/// This is so that we avoid pre-emptively declaring a \c __host__ sibling for
/// a \c __device__ function and causing clashes with \c __host__ handles
/// declared by the source code itself.
/// - Step 2: Using the info accumulated from the previous step, emit the
/// \c __host__ handle for \c __device__ functions that need it inside
/// \c HandleTopLevelDecl.
class EmitHostHandleForDevFuncConsumer : public clang::SemaConsumer {
  clang::Sema *SemaRef{nullptr};

  /// Pre-parse verdict, keyed by stable source location:
  /// \c Synthesize — \c __device__-only functions with no \c __host__ overload
  /// that need one emitted; \c Annotate — existing host-callable counterparts
  /// (user \c __host__ overloads, \c __host__ \c __device__ functions, or
  /// externals) that should be tagged in place.
  llvm::StringSet<> Synthesize;
  llvm::StringSet<> Annotate;

  /// Keys already synthesized, so a function seen twice (separate declaration
  /// and definition) does not get a duplicate host handle.
  llvm::StringSet<> Synthesized;

  /// Set when the pre-parse forwarded a (non-bad-ref) error through the
  /// main diagnostic client.
  bool PrePassFailed{false};

public:
  explicit EmitHostHandleForDevFuncConsumer(clang::CompilerInstance &CI);

  void InitializeSema(clang::Sema &S) override;

  void ForgetSema() override;

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override;
};

} // namespace luthier

#endif
