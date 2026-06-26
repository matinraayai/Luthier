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

namespace clang {
class Decl;
class FunctionDecl;
class Sema;
} // namespace clang

namespace luthier {

class EmitEmptyHostForDevFuncConsumer : public clang::SemaConsumer {
  clang::Sema *SemaRef{nullptr};

  /// Mapping between the device function declarations and their host handle
  llvm::DenseMap<clang::FunctionDecl *, clang::FunctionDecl *>
      DevToHostFuncDecl{};

public:
  EmitEmptyHostForDevFuncConsumer() = default;

  void InitializeSema(clang::Sema &S) override;

  void ForgetSema() override;

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override;

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};

} // namespace luthier

#endif
